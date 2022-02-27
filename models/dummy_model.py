#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import ipdb
import sys
import copy

###
# This is not really a model but simply used as a 
# vessel to have all of my runs look the same.
# So, if our classifier is just some sklearn model,
# we use this to facilitate its training/validation/test.
###

class DummyModel(pl.LightningModule):
    def __init__(self, results_dir='./', lr=0.0001):
        super().__init__()

        self.save_hyperparameters()
        self.dummy_layer = nn.Linear(10, 1)

    def get_callbacks(self, num_epochs):
        return []

    # Add in relevant parameters needed for RAEDE
    def setup(self, stage=None):
        pass

    def forward(self, X):
        return X

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=float(self.hparams['lr']))

    def loss_function(self, recon_x, x):
        return F.mse_loss(recon_x, x)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
