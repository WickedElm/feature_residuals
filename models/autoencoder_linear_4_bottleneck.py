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
import math

class AutoencoderLinear4Bottleneck(pl.LightningModule):
    def __init__(self, in_dims, num_layers, layer_dims, results_dir='./', lr=1e-4, lambda_filter=0.01, dropout_rate=0.0, in_hidden_dim=15):
        super().__init__()

        dropout_rate = 0.0
        self.save_hyperparameters()
        self.in_dims = in_dims

        layer_diff = math.ceil((in_dims - in_hidden_dim) / 4)

        layer1_out = in_dims - layer_diff
        layer2_out = layer1_out - layer_diff
        layer3_out = layer2_out - layer_diff
        layer4_out = in_hidden_dim

        print(f'Layer Dims:  {in_dims}, {layer1_out}, {layer2_out}, {layer3_out}, {in_hidden_dim}')

        self.encoder = nn.Sequential(
            nn.Linear(in_dims, layer1_out),
            nn.ReLU(),
            nn.Linear(layer1_out, layer2_out),
            nn.ReLU(),
            nn.Linear(layer2_out, layer3_out),
            nn.ReLU(),
            nn.Linear(layer3_out, in_hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_hidden_dim, layer3_out),
            nn.ReLU(),
            nn.Linear(layer3_out, layer2_out),
            nn.ReLU(),
            nn.Linear(layer2_out, layer1_out),
            nn.ReLU(),
            nn.Linear(layer1_out, in_dims),
        )

    def get_callbacks(self, num_epochs):
        early_stopping = EarlyStopping(
            monitor='train/loss',
            min_delta=1e-5,
            patience=25,
            mode='min',
            verbose=True,
            stopping_threshold=1e-9,
            check_on_train_epoch_end=False
        )

        #return [early_stopping]
        return []

    # Add in relevant parameters needed for RAEDE
    def setup(self, stage=None):
        if not stage == 'test':
            self.trainer.datamodule.ds_train.original_X = copy.deepcopy(self.trainer.datamodule.ds_train.data_df)
            self.early_stopping_error = 1e-9

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=float(self.hparams['lr']))

    def loss_function(self, recon_x, x):
        feature_loss = list()
        for i in range(self.in_dims):
            if len(recon_x.size()) == 2:
                feature_loss.append(F.mse_loss(recon_x[:,i], x[:,i]))
            else:
                feature_loss.append(F.mse_loss(recon_x[:,:,i], x[:,:,i]))

        return F.mse_loss(recon_x, x), feature_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon = self.forward(x)
        loss, feature_loss = self.loss_function(recon, x)

        self.log('train/loss', loss, on_step=False, on_epoch=True)
        for i in range(len(feature_loss)):
            self.log(f'train/feature_loss_{i}', feature_loss[i], on_step=False, on_epoch=True)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, label = batch
        benign_x = x[label == 0]
        attack_x = x[label == 1]

        recon = self.forward(x)
        loss, feature_loss = self.loss_function(recon, x)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        for i in range(len(feature_loss)):
            self.log(f'val/feature_loss_{i}', feature_loss[i], on_step=False, on_epoch=True)

        benign_recon = self.forward(benign_x)
        benign_loss, benign_feature_loss = self.loss_function(benign_recon, benign_x)
        self.log("val/benign_loss", benign_loss, on_step=False, on_epoch=True)

        for i in range(len(benign_feature_loss)):
            self.log(f'val/benign_feature_loss_{i}', benign_feature_loss[i], on_step=False, on_epoch=True)

        if attack_x.size()[0] > 0:
            attack_recon = self.forward(attack_x)
            attack_loss, attack_feature_loss = self.loss_function(attack_recon, attack_x)

            self.log("val/attack_loss", attack_loss, on_step=False, on_epoch=True)
            for i in range(len(attack_feature_loss)):
                self.log(f'val/attack_feature_loss_{i}', attack_feature_loss[i], on_step=False, on_epoch=True)

            # Calculate automated S thresholds
            threshold_options, feature_threshold_options = self.calculate_threshold_options(benign_loss, attack_loss, benign_feature_loss, attack_feature_loss)

            self.log("val/threshold_options", threshold_options)
            for i in range(len(feature_threshold_options)):
                self.log(f"val/feature_threshold_options_{i}", feature_threshold_options[i])

            return {'val_loss':loss, 'val_benign_loss':benign_loss, 'val_attack_loss':attack_loss}
        else:
            return {'val_loss':loss, 'val_benign_loss':benign_loss}

    def on_validation_epoch_end(self):
        # Get optimized L
        X = torch.tensor(self.trainer.datamodule.ds_train.original_X.iloc[:,1:-1].values)
        L = self.forward(X.float().to(self.device))
        L = L.cpu().detach().numpy()

        # Get our original X
        X = self.trainer.datamodule.ds_train.original_X.iloc[:,1:-1].values

        # alternating project, now project to S
        self.S = X - L
        self.LSO = L + self.S

        # Save off information about L, S, and L + S
        self.loggables = {'L':L, 'S':self.S, 'LSO':self.LSO, 'label':self.trainer.datamodule.ds_train.data_df.iloc[:,-1].values}

    def on_train_end(self):
        # Save off the original data
        original_benign = self.trainer.datamodule.ds_train.original_X.loc[self.trainer.datamodule.ds_train.original_X.label == 0].iloc[:,1:-1]
        original_benign = original_benign.iloc[0:100,:]
        original_benign.to_csv(f'{self.hparams["results_dir"]}/train_original_benign.csv')

        # Save off validation set reconstruction data 
        ldf = pd.DataFrame(self.loggables['L'], columns=self.trainer.datamodule.ds_train.data_df.columns[1:-1])
        ldf['label'] = self.loggables['label']
        ldf.loc[ldf.label == 0].iloc[0:100,:].to_csv(f'{self.hparams["results_dir"]}/train_l_benign.csv', index=False)

        sdf = pd.DataFrame(self.loggables['S'], columns=self.trainer.datamodule.ds_train.data_df.columns[1:-1])
        sdf['label'] = self.loggables['label']
        sdf.loc[sdf.label == 0].iloc[0:100,:].to_csv(f'{self.hparams["results_dir"]}/train_s_benign.csv', index=False)

        lsodf = pd.DataFrame(self.loggables['LSO'], columns=self.trainer.datamodule.ds_train.data_df.columns[1:-1])
        lsodf['label'] = self.loggables['label']
        lsodf.loc[lsodf.label == 0].iloc[0:100,:].to_csv(f'{self.hparams["results_dir"]}/train_lso_benign.csv', index=False)

        # Plot LSO Comparison
        vmax = self.trainer.datamodule.ds_train.original_X.iloc[:,1:-1].max().max()
        s_vmax = sdf.iloc[0:100,:-1].max().max()

        fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True)
        plt.sca(axarr[0])
        
        plt.imshow(original_benign, vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        plt.ylabel('Original')
        plt.xlabel('Benign Network Flows')
        plt.tight_layout()
        
        plt.sca(axarr[1])
        plt.imshow(ldf.loc[ldf.label == 0].iloc[0:100,:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('L')
        plt.tight_layout()
        
        plt.sca(axarr[2])
        plt.imshow(sdf.loc[sdf.label == 0].iloc[0:100,:-1], vmin=-s_vmax, vmax=s_vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('S')
        plt.tight_layout()
        
        # Save plot and close figure
        plt.savefig(f'{self.hparams["results_dir"]}/train_lso_comparison.png')
        wandb.log({'train/lso_comparison':plt})
        plt.close()

        # Perform some comparisons
        X = self.trainer.datamodule.ds_train.original_X.iloc[:,1:-1].values

        # Are L or S close to zero?
        zero_df = np.zeros_like(self.loggables['L'])
        LZ = np.allclose(self.loggables['L'], zero_df, rtol=1e-3, atol=1e-5)
        SZ = np.allclose(self.loggables['S'], zero_df, rtol=1e-3, atol=1e-5)
        XLSO = np.allclose(X, self.loggables['LSO'], rtol=1e-2, atol=1e-2)
        print(f'LS = {LZ}, SZ = {SZ}, XLSO = {XLSO}')
        print(f'Threshold Options:  {self.threshold_options}')

    def test_step(self, batch, batch_idx):
        pass

    def calculate_threshold_options(self, benign_loss, attack_loss, benign_feature_loss, attack_feature_loss):
        high_threshold = np.max([benign_loss, attack_loss])
        low_threshold = np.min([benign_loss, attack_loss])
        fifty_percent_threshold = (high_threshold + low_threshold) / 2.
        seventy_five_percent_threshold = (high_threshold + low_threshold) * 0.75
        twenty_five_percent_threshold = (high_threshold + low_threshold) * 0.25

        # Store options for later
        self.threshold_options = [high_threshold, seventy_five_percent_threshold, fifty_percent_threshold, twenty_five_percent_threshold, low_threshold]
        self.threshold_names = ['high', '75_percent', '50_percent', '25_percent', 'low']

        ###
        # Feature Thresholds
        ###
        feature_high_threshold = list()
        feature_low_threshold = list()
        feature_fifty_percent_threshold = list()
        feature_seventy_five_percent_threshold = list()
        feature_twenty_five_percent_threshold = list()
        for i in range(len(benign_feature_loss)):
            high_threshold = np.max([benign_feature_loss[i], attack_feature_loss[i]])
            low_threshold = np.min([benign_feature_loss[i], attack_feature_loss[i]])
            fifty_percent_threshold = (high_threshold + low_threshold) / 2.
            seventy_five_percent_threshold = (high_threshold + low_threshold) * 0.75
            twenty_five_percent_threshold = (high_threshold + low_threshold) * 0.25

            feature_high_threshold.append(high_threshold)
            feature_low_threshold.append(low_threshold)
            feature_fifty_percent_threshold.append(fifty_percent_threshold)
            feature_seventy_five_percent_threshold.append(seventy_five_percent_threshold)
            feature_twenty_five_percent_threshold.append(twenty_five_percent_threshold)

        self.feature_threshold_options = [feature_high_threshold, feature_low_threshold, feature_fifty_percent_threshold, feature_seventy_five_percent_threshold, feature_twenty_five_percent_threshold]
        self.feature_threshold_names = ['high_features', '75_percent_features', '50_percent_features', '25_percent_features', 'low_features']

        return self.threshold_options, self.feature_threshold_options

    def on_save_checkpoint(self, checkpoint):
        checkpoint['threshold_options'] = self.threshold_options
        checkpoint['threshold_names'] = self.threshold_names
        checkpoint['feature_threshold_options'] = self.feature_threshold_options
        checkpoint['feature_threshold_names'] = self.feature_threshold_names

    def on_load_checkpoint(self, checkpoint):
        self.threshold_options = checkpoint['threshold_options']
        self.threshold_names = checkpoint['threshold_names']
        self.feature_threshold_options = checkpoint['feature_threshold_options']
        self.feature_threshold_names = checkpoint['feature_threshold_names']

