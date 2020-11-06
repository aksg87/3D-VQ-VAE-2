import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union, Tuple

import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from monai import transforms

from model import VQVAE
from utils import CTScanDataset


class CTDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=64, train_frac=0.95, num_workers=6):
        super().__init__()
        assert 0 <= train_frac <= 1

        self.path = path
        self.train_frac = train_frac
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        # transform
        key, min_val, max_val, scale_val = 'img', -1500, 3000, 1000

        transform = transforms.Compose([
            transforms.AddChannel(),
            transforms.ThresholdIntensity(threshold=max_val, cval=max_val, above=False),
            transforms.ThresholdIntensity(threshold=min_val, cval=min_val, above=True),
            transforms.ScaleIntensity(minv=None, maxv=None, factor=(-1 + 1/scale_val)),
            transforms.ShiftIntensity(offset=1),
            transforms.SpatialPad(spatial_size=(512, 512, 128), mode='constant'),
            transforms.RandSpatialCrop(roi_size=(512, 512, 128), random_size=False),
            transforms.ToTensor()
        ])

        dataset = CTScanDataset(self.path, transform=transform, spacing=(0.976, 0.976, 3))

        train_len = int(len(dataset) * self.train_frac)
        val_len = len(dataset) - train_len

        # train/val split
        train_split, val_split = random_split(dataset, [train_len, val_len])

        # assign to use in dataloaders
        self.train_dataset = train_split
        self.train_len = train_len
        self.train_batch_size = self.batch_size

        self.val_dataset = val_split
        self.val_len = val_len
        self.val_batch_size = self.batch_size


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, drop_last=True)


def main(args):
    pl.trainer.seed_everything(seed=42)

    datamodule = CTDataModule(path=args.dataset_path, batch_size=args.batch_size, num_workers=6)

    model = VQVAE(args)
    if args.checkpoint_path != '':
        model = model.load_from_checkpoint(args.checkpoint_path)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=3, monitor='val_loss_mean')
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        gpus="-1",
        auto_select_gpus=True,
        distributed_backend='ddp',
        benchmark=True,
        num_nodes=args.num_nodes,

        accumulate_grad_batches=args.accumulate_grad_batches,

        terminate_on_nan=True,

        profiler=None,

        checkpoint_callback=checkpoint_callback,
        log_every_n_steps=50,
        val_check_interval=100,
        flush_logs_every_n_steps=100,
        weights_summary='full',

        # limit_val_batches=0,
        # overfit_batches=1,

        callbacks=[lr_logger],
    )

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-path", default='')
    parser.add_argument("dataset_path", type=Path)

    args = parser.parse_args()


    main(args)