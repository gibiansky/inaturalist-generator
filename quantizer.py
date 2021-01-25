"""
A model for quantizing images to integer codes.
"""
import time
import random
from typing import List, Any
import os

import click
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image  # type: ignore
from pytorch_lightning import loggers as pl_loggers

from dataloader import ImageDataModule

RGB_CHANNELS: int = 3
IMAGE_HEIGHT: int = 256
IMAGE_WIDTH: int = 256


class Downsample(nn.Module):
    """
    Learned downsampling.
    """

    def __init__(self, ratio: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ratio, stride=ratio),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Resnet(nn.Module):
    """
    Conv resnet for a single scale.
    """

    def __init__(self, n_layers: int, channels: int, kernel_size: int) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.skip: nn.Module = nn.Conv2d(channels, channels, 1)

        layers: List[nn.Module] = []
        for i in range(n_layers):
            layers.extend(
                (
                    nn.Conv2d(
                        channels, channels, kernel_size, padding=(kernel_size - 1) // 2
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                )
            )
        self.resnet: nn.Module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x) + self.skip(x)


class Encoder(nn.Module):
    """The image encoder."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        channels = config.encoder.init_channels
        layers = [
            nn.Conv2d(
                RGB_CHANNELS,
                channels,
                config.encoder.init_kernel_size,
                padding=(config.encoder.init_kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        ]
        for downsample in config.encoder.downsampling:
            out_channels = channels * downsample
            layers.extend(
                (
                    Downsample(downsample, channels, out_channels),
                    Resnet(
                        config.encoder.layers_per_scale,
                        out_channels,
                        config.encoder.kernel_size,
                    ),
                )
            )
            channels = out_channels
        self.model: nn.Module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Quantizer(nn.Module):
    """Gumbel-softmax quantization."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        in_channels = int(
            np.prod(config.encoder.downsampling) * config.encoder.init_channels
        )
        self.predictor: nn.Module = nn.Conv2d(
            in_channels, config.quantizers.classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = self.predictor(x)
        return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=1)


class Upsample(nn.Module):
    """
    Learned upsampling.
    """

    def __init__(self, ratio: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model: nn.Module = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=ratio, stride=ratio
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decoder(nn.Module):
    """The image decoder."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        input_channels = config.quantizers.count * config.quantizers.classes
        channels = config.decoder.init_channels
        layers: List[nn.Module] = [
            nn.Conv2d(input_channels, channels, kernel_size=1),
        ]
        for upsample in config.decoder.upsampling:
            assert channels % upsample == 0
            out_channels = channels // upsample
            layers.extend(
                (
                    Upsample(upsample, channels, out_channels),
                    Resnet(
                        config.decoder.layers_per_scale,
                        out_channels,
                        config.decoder.kernel_size,
                    ),
                )
            )
            channels = out_channels
        layers.extend(
            (
                nn.Conv2d(
                    channels,
                    RGB_CHANNELS,
                    config.decoder.final_kernel_size,
                    padding=(config.decoder.final_kernel_size - 1) // 2,
                ),
                nn.Sigmoid(),
            )
        )
        self.model: nn.Module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ImageQuantizer(pl.LightningModule):
    """The image quantizer model."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.encoder: nn.Module = Encoder(config)
        self.quantizers: nn.ModuleList = nn.ModuleList(
            [Quantizer(config) for _ in range(config.quantizers.count)]
        )
        self.decoder: nn.Module = Decoder(config)
        self.config: Any = config
        self.wrote_true_images = set()

    def get_temperature(self) -> float:
        iteration = self.global_step
        config = self.config.quantizers
        if iteration < config.start_anneal:
            return config.max_temp
        if iteration > config.stop_anneal:
            return config.min_temp
        n_iters = config.stop_anneal - config.start_anneal
        frac = 1.0 - (iteration - config.start_anneal) / n_iters
        return config.min_temp + frac * (config.max_temp - config.min_temp)

    def forward(self, x: torch.Tensor, temperature: float) -> List[torch.Tensor]:  # type: ignore
        encoded = self.encoder(x)
        return [q(encoded, temperature) for q in self.quantizers]

    def autoencode(self, batch: torch.Tensor) -> torch.Tensor:
        temperature = self.get_temperature()
        quantized = self.forward(batch, temperature)
        hidden = torch.cat(quantized, dim=1)
        decoded = self.decoder(hidden)
        return decoded

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        decoded = self.autoencode(batch)
        loss = 100.0 * F.mse_loss(decoded, batch)
        self.log("train_loss", loss)
        self.log("temperature", self.get_temperature())
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        decoded = self.autoencode(batch)
        loss = F.mse_loss(decoded, batch)
        self.log("valid_loss", loss)

        if batch_idx % 10 != 0:
            return loss

        idx = random.Random(batch_idx).randint(0, batch.shape[0] - 1)
        true = batch[idx, :, :, :]
        pred = decoded[idx, :, :, :]
        self.logger.experiment.add_image(
            f"valid/{batch_idx}_{idx}_pred", pred, self.global_step
        )
        if batch_idx not in self.wrote_true_images:
            self.wrote_true_images.add(batch_idx)
            self.logger.experiment.add_image(
                f"valid/{batch_idx}_{idx}_true", true, self.global_step
            )

        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.config.adam.lr)


@click.group()
def main():
    """Quantizer command."""


@main.command("train")
@click.argument("config_path")
@click.option("--data", "data_path", default="data", help="Path to dataset")
@click.option("--checkpoint", help="Path to checkpoint")
@click.option("--save", "save_dir", required=True, help="Where to save")
@click.argument("options", nargs=-1)
def train_command(config_path, data_path, checkpoint, save_dir, options):
    """Train the quantizer."""
    config = OmegaConf.load(config_path)
    config.merge_with_dotlist(options)
    model = ImageQuantizer(config)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)["state_dict"])

    data_module = ImageDataModule(config, data_path, IMAGE_HEIGHT, IMAGE_WIDTH)

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    trainer = pl.Trainer(val_check_interval=0.05, gpus=1,
                         default_root_dir=save_dir, logger=tb_logger)
    trainer.fit(model, data_module)


@main.command("data-loading-benchmark")
@click.argument("config_path")
@click.argument("data_path")
def data_loading_benchmark_command(config_path, data_path):
    """Train the quantizer."""
    config = OmegaConf.load(config_path)
    data_module = ImageDataModule(config, data_path, IMAGE_HEIGHT, IMAGE_WIDTH)
    start_time = time.time()
    for i, _batch in enumerate(data_module.train_dataloader()):
        elapsed = time.time() - start_time
        print(f"Avg[i={i}]: {elapsed / (i + 1)}")


@main.command("run")
@click.argument("config_path")
@click.argument("image_path")
@click.option("--checkpoint", required=True, help="Path to checkpoint")
@click.option("--output", required=True, help="Path to write to")
def run_command(config_path, image_path, checkpoint, output):
    """Run the quantizer."""
    config = OmegaConf.load(config_path)
    model = ImageQuantizer(config)
    model.load_state_dict(torch.load(checkpoint)["state_dict"])

    image = Image.open(image_path).convert("RGB")
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop([IMAGE_HEIGHT, IMAGE_WIDTH]),
            torchvision.transforms.ToTensor(),
        ]
    )

    with torch.no_grad():
        image_tensor = transforms(image)
        predicted = model.autoencode(image.unsqueeze(0)).squeeze(0)
        predicted = predicted.cpu().numpy()

    out_image = Image.fromarray(predicted * 255, mode="RGB")
    out_image.save(output)


if __name__ == "__main__":
    main()
