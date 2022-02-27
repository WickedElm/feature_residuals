#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import torch
import torchmetrics
from metrics.utilities import *
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
import math

class ClfMlp(pl.LightningModule):
    def __init__(self, in_dims, num_layers, layer_dims, feature_transformer, threshold_name, results_dir='./', experiment='default', lr=1.0, dropout_rate=0.99):
        super().__init__()

        self.save_hyperparameters()

        self.rae = None
        self.feature_transformer = feature_transformer
        self.results_dir = results_dir
        self.experiment = experiment
        self.metrics = dict()
        self.threshold_name = threshold_name
        self.s_threshold = None

        self.loss_fn = None

    def get_callbacks(self, num_epochs):
        early_stopping_train_loss = EarlyStopping(
            monitor='classifier/train/loss',
            min_delta=1e-5,
            patience=25,
            mode='min',
            verbose=True,
            stopping_threshold=1e-6,
            check_on_train_epoch_end=False
        )

        #return [early_stopping_train_loss]
        return []

    def setup(self, stage=None):
        if not stage == 'test':
            # Use mixed data
            self.trainer.datamodule.ds_train.data_df = copy.deepcopy(self.trainer.datamodule.ds_train.all_training_data)

            # Weight our classes
            label = torch.tensor(self.trainer.datamodule.ds_train.data_df.iloc[:,-1].values).float().to(self.rae.device)
            num_attacks = label.sum()
            num_benign = label.shape[0] - num_attacks

            # Set loss function
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([num_benign / num_attacks]))

            # Use all training data to determine thresholds
            self.recalculate_training_thresholds()

            # Construct layers using transformed data dimensions
            orig_x, y = self.trainer.datamodule.ds_train.__getitem__(0)
            x, _ = self.feature_transformer.torch_transform(None, self.rae, orig_x, y, self.s_threshold)
            in_dims = x.shape[1]
            print('LEWBUG:', in_dims)
            layer1_out_dims = math.ceil(in_dims * 0.7)
            hidden_layer_out = math.ceil(in_dims * 0.3)

            self.hidden_layers = nn.Sequential(
                nn.Linear(in_dims, layer1_out_dims),
                nn.ReLU(),
                nn.Linear(layer1_out_dims, hidden_layer_out),
                nn.ReLU()
            )

            self.output_layer = nn.Sequential(
                nn.Linear(hidden_layer_out, 1),
                # Sigmoid activation applied in loss function
            )


    def recalculate_training_thresholds(self):
        # Recalculate the thresholds based on autoencoder reconstruction
        # Using the full training set of data
        # This mimics the autoencoder's validation_step core functionality

        self.training_data = copy.deepcopy(self.trainer.datamodule.ds_train.all_training_data)

        if self.threshold_name == 'no_threshold':
            return

        x = torch.tensor(self.training_data.iloc[:,1:-1].values).float().to(self.rae.device)
        label = torch.tensor(self.training_data.iloc[:,-1].values).float().to(self.rae.device)
        benign_x = x[label == 0]
        attack_x = x[label == 1]

        recon = self.rae.forward(x)
        loss, feature_loss = self.rae.loss_function(recon, x)

        benign_recon = self.rae.forward(benign_x)
        benign_loss, benign_feature_loss = self.rae.loss_function(benign_recon, benign_x)

        attack_recon = self.rae.forward(attack_x)
        attack_loss, attack_feature_loss = self.rae.loss_function(attack_recon, attack_x)

        threshold_options, feature_threshold_options = self.rae.calculate_threshold_options(benign_loss, attack_loss, benign_feature_loss, attack_feature_loss)

        # Update s_threshold to use new threshold
        if 'feature' in self.threshold_name:
            threshold_index = self.rae.feature_threshold_names.index(self.threshold_name)
            self.s_threshold = self.rae.feature_threshold_options[threshold_index]
        else:
            threshold_index = self.rae.threshold_names.index(self.threshold_name)
            self.s_threshold = self.rae.threshold_options[threshold_index]

    def forward(self, X):
        X = self.hidden_layers(X)
        X = self.output_layer(X)
        predictions = torch.round(torch.sigmoid(X))
        return X, torch.tensor(predictions, dtype=int)

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=float(self.hparams['lr']))

    def loss_function(self, y_pred, y):
        return self.loss_fn(y_pred, y.unsqueeze(1))

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, _ = self.feature_transformer.torch_transform(None, self.rae, x, y, self.s_threshold)
        y_pred_prob, y_pred = self.forward(x)
        loss = self.loss_function(y_pred_prob, y)

        self.log('classifier/train/loss', loss, on_step=False, on_epoch=True)
        return {'loss':loss, 'y_pred':y_pred, 'y_true':y}

    def training_epoch_end(self, training_step_outputs):
        self.log_epoch_end(f'classifier/train_{self.threshold_name}/', training_step_outputs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, _ = self.feature_transformer.torch_transform(None, self.rae, x, y, self.s_threshold)
        y_pred_prob, y_pred = self.forward(x)
        loss = self.loss_function(y_pred_prob, y)

        self.log(f'classifier/val_{self.threshold_name}/loss', loss, on_step=False, on_epoch=True)
        return {'val_loss':loss, 'y_pred':y_pred, 'y_true':y}

    def validation_epoch_end(self, val_step_outputs):
        self.log_epoch_end(f'classifier/val_{self.threshold_name}/', val_step_outputs)

    def on_train_end(self):
        pass
        #EXPERIMENTS_DB = './output/experiments_db'

        ## Get confusion matrix
        #cm, cm_norm = plot_confusion_matrix(
        #    predicted_labels=self.metrics['classifier/val/predictions'],
        #    true_labels=self.metrics['classifier/val/truth'],
        #    metrics_dir=self.results_dir,
        #    is_training=False,
        #    title=f'{self.trainer.datamodule.dataset_name} Validation Set',
        #    target_names=['benign','attack']
        #)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, _ = self.feature_transformer.torch_transform(None, self.rae, x, y, self.s_threshold)
        y_pred_prob, y_pred = self.forward(x)
        loss = self.loss_function(y_pred_prob, y)
        self.log(f'classifier/test_{self.threshold_name}/loss', loss, on_step=False, on_epoch=True)
        return {'test_loss':loss, 'y_pred':y_pred, 'y_true':y}

    def test_epoch_end(self, test_step_outputs):
        self.log_epoch_end(f'classifier/test_{self.threshold_name}/', test_step_outputs)

    def log_epoch_end(self, epoch_type, outputs):
        # Reconstruct y_pred and y_true
        predictions = torch.cat([out['y_pred'] for out in outputs]).squeeze(1).detach().numpy()
        truth = torch.tensor(torch.cat([out['y_true'] for out in outputs]), dtype=int).detach().numpy()

        self.metrics[f'{epoch_type}predictions'] = predictions
        self.metrics[f'{epoch_type}truth'] = truth

        # Obtain and log metrics
        print(f'{epoch_type}:')

        # Collect metrics
        self.metrics[f'{epoch_type}prec'] = precision(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            log_to_disk=False
        )
        
        self.metrics[f'{epoch_type}rec'] = recall(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            log_to_disk=False
        )
        
        self.metrics[f'{epoch_type}f1'] = f1_score(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            log_to_disk=False
        )
        
        # Log our metrics
        self.log(f'{epoch_type}{self.feature_transformer.__class__.__name__}_precision', self.metrics[f'{epoch_type}prec'], on_step=False, on_epoch=True)
        self.log(f'{epoch_type}{self.feature_transformer.__class__.__name__}_recall', self.metrics[f'{epoch_type}rec'], on_step=False, on_epoch=True)
        self.log(f'{epoch_type}{self.feature_transformer.__class__.__name__}_f1-score', self.metrics[f'{epoch_type}f1'], on_step=False, on_epoch=True)
