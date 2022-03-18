#!/usr/bin/env python

import argparse
import importlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from metrics.utilities import *
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import ipdb
import sys
import dataset.Dataset
import glob
import re
import copy
import math
import pickle
import socket

def list_models():
    # Get all files in models directory
    model_files = glob.glob('./models/*.py')

    print('')
    print('Models Available: (--model)')
    print('')

    # Get all classes in those files
    for mf in model_files:
        if '__init__' in mf:
            continue
        
        # Get module name from file name
        _, file_name = os.path.split(mf)
        file_name = file_name.replace('.py', '')

        # Get agent classes from file
        with open(mf, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if re.match(r'^class', line):
                model_class_name = (re.findall(r'^class\s+(\w+)\(.*', line))[0]
                print(f'        - {file_name}.{model_class_name}')
    print('')

class NetflowToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample).float()

class StandardNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, 1:-1].shape

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, 1:-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        return flow, label

class NetflowConferenceDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./', batch_size=32, total_rows_threshold=500000, weighted=False, load_from_disk=False, prefix='', reserve_type=''):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.total_rows_threshold = total_rows_threshold
        self.load_from_disk = load_from_disk
        self.prefix = prefix
        self.reserve_type = reserve_type
        self.load_data_path = './tmp_ds_data'
        self.transform = transforms.Compose([
            NetflowToTensor()
        ])
        self.data_loaded = False

        self.weighted = weighted
        self.sampler = None

    def prepare_data(self):
        # TODO:  Download data is needed
        pass

    def setup(self, stage=None):

        if not self.data_loaded:
            # NOTE:  May need to think about this if more than 1 GPU is ever used
            # Load our dataset using data_path
            ds = dataset.Dataset.Dataset.load(self.data_path)
            self.dataset_name = ds.name
            ds.sample_data(total_rows_threshold=self.total_rows_threshold)

            # Insert new processors/updates here
            ds.append_processor(dataset.DatasetProcessors.AddBytesPerSecondFeatures(), process=False)
            ds.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_bytes', 'total_source_bytes', 'total_destination_bytes', 'total_bytes_per_second', 'source_bytes_per_second', 'destination_bytes_per_second'], function=math.log), process=False)
            ds.cols_to_normalize.append('total_bytes_per_second')
            ds.cols_to_normalize.append('destination_bytes_per_second')
            ds.cols_to_normalize.append('source_bytes_per_second')

            ds.append_processor(dataset.DatasetProcessors.AddPacketsPerSecondFeatures(), process=False)
            ds.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_packets', 'source_packets', 'destination_packets', 'packets_per_second', 'source_packets_per_second', 'destination_packets_per_second'], function=math.log), process=False)
            ds.cols_to_normalize.append('packets_per_second')
            ds.cols_to_normalize.append('destination_packets_per_second')
            ds.cols_to_normalize.append('source_packets_per_second')

            ds.append_processor(dataset.DatasetProcessors.SortColumnsAlphabetically(), process=False)
            ds.append_processor(dataset.DatasetProcessors.MakeTimestampColumnFirst(), process=False)

            if self.load_from_disk:
                if not ds.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type):
                    print('Generating data splits.')
                    ds.perform_processing()
                    ds.save_split_to_disk(self.load_data_path, self.prefix, 'rae_lambda_tuning', 0.67, 0.1, 0.2)
                    ds.save_split_to_disk(self.load_data_path, self.prefix, 'rae_classifier_tuning', 0.5, 0.1, 0.2)
                    ds.save_split_to_disk(self.load_data_path, self.prefix, 'rae_full_loop_testing', 1.0, 0.1, 0.2)
                    ds.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type)
            else:
                ds.perform_processing()
                ds.default_stratified_split(split_point_1=0.15, split_point_2=0.3)

            ds.save_indices()
            ds.write_indices_to_disk(opt.results_dir, f'{opt.save_prefix}_{opt.experiment}')
            ds.normalize_columns_zero_max(target_cols=ds.cols_to_normalize)
            ds.print_dataset_info()

            self.data_loaded = True

            # Construct training and validation sets
            self.ds_train = StandardNetflowDataset(ds.training_data, self.transform)
            self.ds_val = StandardNetflowDataset(ds.validation_data, self.transform)
            self.ds_test = StandardNetflowDataset(ds.test_data, self.transform)

        if stage == 'train' or stage == None:
            self.dims = tuple(self.ds_train.dims)

        if stage == 'test':
            self.dims = tuple(self.ds_test.dims)

    def train_dataloader(self):
        if self.weighted:
            print('LEWBUG:  Using weighted sampler')
            if self.sampler is None:
                print('LEWBUG:  Creating sampler')
                # Number of samples
                num_samples = self.ds_train.data_df.shape[0]
                benign_samples = self.ds_train.data_df[self.ds_train.data_df.label == 0].shape[0]
                attack_samples = self.ds_train.data_df[self.ds_train.data_df.label == 1].shape[0]
                print(f'Benign Samples:  {benign_samples}')
                print(f'Attack Samples:  {attack_samples}')
                class_weights = [1. / benign_samples, 1. / attack_samples]
                sample_weights = self.ds_train.data_df.iloc[:,-1]
                sample_weights = sample_weights.apply(lambda x: class_weights[0] if x == 0 else class_weights[1])
                sample_weights = torch.tensor(sample_weights.values)

                self.sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
            return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8, sampler=self.sampler)
        else:
            return DataLoader(self.ds_train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
            return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8)

class NetflowConferenceBenignOnlyTrainingDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./', batch_size=32, total_rows_threshold=500000, weighted=False, load_from_disk=False, prefix='', reserve_type=''):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.total_rows_threshold = total_rows_threshold
        self.load_from_disk = load_from_disk
        self.prefix = prefix
        self.reserve_type = reserve_type
        self.load_data_path = './tmp_ds_data'
        self.transform = transforms.Compose([
            NetflowToTensor()
        ])
        self.data_loaded = False

        self.weighted = weighted
        self.sampler = None

    def prepare_data(self):
        # TODO:  Download data is needed
        pass

    def setup(self, stage=None):

        if not self.data_loaded:
            # NOTE:  May need to think about this if more than 1 GPU is ever used
            # Load our dataset using data_path
            ds = dataset.Dataset.Dataset.load(self.data_path)
            self.dataset_name = ds.name
            ds.sample_data(total_rows_threshold=self.total_rows_threshold)

            # Insert new processors/updates here
            ds.append_processor(dataset.DatasetProcessors.AddBytesPerSecondFeatures(), process=False)
            ds.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_bytes', 'total_source_bytes', 'total_destination_bytes', 'total_bytes_per_second', 'source_bytes_per_second', 'destination_bytes_per_second'], function=math.log), process=False)
            ds.cols_to_normalize.append('total_bytes_per_second')
            ds.cols_to_normalize.append('destination_bytes_per_second')
            ds.cols_to_normalize.append('source_bytes_per_second')

            ds.append_processor(dataset.DatasetProcessors.AddPacketsPerSecondFeatures(), process=False)
            ds.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_packets', 'source_packets', 'destination_packets', 'packets_per_second', 'source_packets_per_second', 'destination_packets_per_second'], function=math.log), process=False)
            ds.cols_to_normalize.append('packets_per_second')
            ds.cols_to_normalize.append('destination_packets_per_second')
            ds.cols_to_normalize.append('source_packets_per_second')

            ds.append_processor(dataset.DatasetProcessors.SortColumnsAlphabetically(), process=False)
            ds.append_processor(dataset.DatasetProcessors.MakeTimestampColumnFirst(), process=False)

            if self.load_from_disk:
                if not ds.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type):
                    print('Generating data splits.')
                    ds.perform_processing()
                    ds.save_split_to_disk(self.load_data_path, self.prefix, 'ae_tuning', 0.5, 0.1, 0.2)
                    ds.save_split_to_disk(self.load_data_path, self.prefix, 'full_loop_testing', 1.0, 0.1, 0.2)
                    #ds.save_split_to_disk(self.load_data_path, self.prefix, 'rae_lambda_tuning', 0.67, 0.1, 0.2)
                    #ds.save_split_to_disk(self.load_data_path, self.prefix, 'rae_classifier_tuning', 0.5, 0.1, 0.2)
                    #ds.save_split_to_disk(self.load_data_path, self.prefix, 'rae_full_loop_testing', 1.0, 0.1, 0.2)
                    ds.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type)
            else:
                ds.perform_processing()
                ds.default_stratified_split(split_point_1=0.15, split_point_2=0.3)

            ds.save_indices()
            ds.write_indices_to_disk(opt.results_dir, f'{opt.save_prefix}_{opt.experiment}')

            # Shuffle our data
            ds.training_data = ds.training_data.sample(frac=1)
            ds.validation_data = ds.validation_data.sample(frac=1)
            ds.test_data = ds.test_data.sample(frac=1)

            # Only use benign samples for training data for the RAE
            # We save all data for usage later if needed
            all_training_data = copy.deepcopy(ds.training_data)
            ds.training_data = ds.training_data.loc[ds.training_data.label == 0]

            ds.normalize_columns_zero_max(target_cols=ds.cols_to_normalize)

            # This needs to be done based on our benign only training data to avoid snooping
            all_training_data[ds.cols_to_normalize] = all_training_data[ds.cols_to_normalize] / ds.scaler
            ds.print_dataset_info()

            self.data_loaded = True

            # Construct training and validation sets
            self.ds_train = StandardNetflowDataset(ds.training_data, self.transform)
            self.ds_train.all_training_data = all_training_data
            self.ds_val = StandardNetflowDataset(ds.validation_data, self.transform)
            self.ds_test = StandardNetflowDataset(ds.test_data, self.transform)

        if stage == 'train' or stage == None:
            self.dims = tuple(self.ds_train.dims)

        if stage == 'test':
            self.dims = tuple(self.ds_test.dims)

    def train_dataloader(self):
        if self.weighted:
            print('LEWBUG:  Using weighted sampler')
            if self.sampler is None:
                print('LEWBUG:  Creating sampler')
                # Number of samples
                num_samples = self.ds_train.data_df.shape[0]
                benign_samples = self.ds_train.data_df[self.ds_train.data_df.label == 0].shape[0]
                attack_samples = self.ds_train.data_df[self.ds_train.data_df.label == 1].shape[0]
                print(f'Benign Samples:  {benign_samples}')
                print(f'Attack Samples:  {attack_samples}')
                class_weights = [1. / benign_samples, 1. / attack_samples]
                sample_weights = self.ds_train.data_df.iloc[:,-1]
                sample_weights = sample_weights.apply(lambda x: class_weights[0] if x == 0 else class_weights[1])
                sample_weights = torch.tensor(sample_weights.values)

                self.sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
            return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8, sampler=self.sampler)
        else:
            return DataLoader(self.ds_train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
            return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8)

class NetflowConferenceCICIDS2017BenignOnlyTrainingDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./', batch_size=32, total_rows_threshold=500000, weighted=False, load_from_disk=False, prefix='', reserve_type=''):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.total_rows_threshold = total_rows_threshold
        self.load_from_disk = load_from_disk
        self.prefix = prefix
        self.reserve_type = reserve_type
        self.load_data_path = './tmp_ds_data'
        self.transform = transforms.Compose([
            NetflowToTensor()
        ])
        self.data_loaded = False

        self.weighted = weighted
        self.sampler = None

    def prepare_data(self):
        # TODO:  Download data is needed
        pass

    def setup(self, stage=None):

        if not self.data_loaded:
            # NOTE:  May need to think about this if more than 1 GPU is ever used
            # Load our dataset using data_path
            ds = dataset.Dataset.Dataset.load(self.data_path)
            ds_monday = dataset.Dataset.Dataset.load('./datasets/cicids2017/sf_monday/sf_monday.pkl')
            self.dataset_name = ds.name
            ds.sample_data(total_rows_threshold=self.total_rows_threshold)
            ds_monday.sample_data(total_rows_threshold=self.total_rows_threshold)

            # Insert new processors/updates here
            ds.append_processor(dataset.DatasetProcessors.AddBytesPerSecondFeatures(), process=False)
            ds.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_bytes', 'total_source_bytes', 'total_destination_bytes', 'total_bytes_per_second', 'source_bytes_per_second', 'destination_bytes_per_second'], function=math.log), process=False)
            ds.cols_to_normalize.append('total_bytes_per_second')
            ds.cols_to_normalize.append('destination_bytes_per_second')
            ds.cols_to_normalize.append('source_bytes_per_second')

            ds.append_processor(dataset.DatasetProcessors.AddPacketsPerSecondFeatures(), process=False)
            ds.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_packets', 'source_packets', 'destination_packets', 'packets_per_second', 'source_packets_per_second', 'destination_packets_per_second'], function=math.log), process=False)
            ds.cols_to_normalize.append('packets_per_second')
            ds.cols_to_normalize.append('destination_packets_per_second')
            ds.cols_to_normalize.append('source_packets_per_second')

            ds.append_processor(dataset.DatasetProcessors.SortColumnsAlphabetically(), process=False)
            ds.append_processor(dataset.DatasetProcessors.MakeTimestampColumnFirst(), process=False)
            
            # Now for Monday
            ds_monday.append_processor(dataset.DatasetProcessors.AddBytesPerSecondFeatures(), process=False)
            ds_monday.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_bytes', 'total_source_bytes', 'total_destination_bytes', 'total_bytes_per_second', 'source_bytes_per_second', 'destination_bytes_per_second'], function=math.log), process=False)
            ds_monday.cols_to_normalize.append('total_bytes_per_second')
            ds_monday.cols_to_normalize.append('destination_bytes_per_second')
            ds_monday.cols_to_normalize.append('source_bytes_per_second')

            ds_monday.append_processor(dataset.DatasetProcessors.AddPacketsPerSecondFeatures(), process=False)
            ds_monday.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_packets', 'source_packets', 'destination_packets', 'packets_per_second', 'source_packets_per_second', 'destination_packets_per_second'], function=math.log), process=False)
            ds_monday.cols_to_normalize.append('packets_per_second')
            ds_monday.cols_to_normalize.append('destination_packets_per_second')
            ds_monday.cols_to_normalize.append('source_packets_per_second')

            ds_monday.append_processor(dataset.DatasetProcessors.SortColumnsAlphabetically(), process=False)
            ds_monday.append_processor(dataset.DatasetProcessors.MakeTimestampColumnFirst(), process=False)

            if self.load_from_disk:
                if not ds_monday.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type):
                    print('Generating Monday data splits.')
                    ds_monday.perform_processing()
                    ds_monday.save_split_to_disk(self.load_data_path, self.prefix, 'ae_tuning', 0.5, 0.01, 0.02)
                    ds_monday.save_split_to_disk(self.load_data_path, self.prefix, 'full_loop_testing', 1.0, 0.01, 0.02)
                    ds_monday.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type)

                if not ds.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type):
                    print('Generating data splits.')
                    ds.perform_processing()
                    ds.save_split_to_disk(self.load_data_path, self.prefix, 'ae_tuning', 0.5, 0.1, 0.2)
                    ds.save_split_to_disk(self.load_data_path, self.prefix, 'full_loop_testing', 1.0, 0.1, 0.2)
                    ds.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type)
            else:
                ds.perform_processing()
                ds_monday.perform_processing()
                ds.default_stratified_split(split_point_1=0.15, split_point_2=0.3)
                ds.default_stratified_split(split_point_1=0.01, split_point_2=0.01)

            ds.save_indices()
            ds.write_indices_to_disk(opt.results_dir, f'{opt.save_prefix}_{opt.experiment}')

            # Shuffle our data
            ds.training_data = ds.training_data.sample(frac=1)
            ds.validation_data = ds.validation_data.sample(frac=1)
            ds.test_data = ds.test_data.sample(frac=1)

            # Only use Monday benign samples for training data for the RAE
            # We save all data for usage later if needed
            all_training_data = copy.deepcopy(ds.training_data)
            ds.training_data = copy.deepcopy(ds_monday.training_data)

            ds.normalize_columns_zero_max(target_cols=ds.cols_to_normalize)

            # This needs to be done based on our benign only training data to avoid snooping
            all_training_data[ds.cols_to_normalize] = all_training_data[ds.cols_to_normalize] / ds.scaler
            ds.print_dataset_info()

            self.data_loaded = True

            # Construct training and validation sets
            self.ds_train = StandardNetflowDataset(ds.training_data, self.transform)
            self.ds_train.all_training_data = all_training_data
            self.ds_val = StandardNetflowDataset(ds.validation_data, self.transform)
            self.ds_test = StandardNetflowDataset(ds.test_data, self.transform)

        if stage == 'train' or stage == None:
            self.dims = tuple(self.ds_train.dims)

        if stage == 'test':
            self.dims = tuple(self.ds_test.dims)

    def train_dataloader(self):
        if self.weighted:
            print('LEWBUG:  Using weighted sampler')
            if self.sampler is None:
                print('LEWBUG:  Creating sampler')
                # Number of samples
                num_samples = self.ds_train.data_df.shape[0]
                benign_samples = self.ds_train.data_df[self.ds_train.data_df.label == 0].shape[0]
                attack_samples = self.ds_train.data_df[self.ds_train.data_df.label == 1].shape[0]
                print(f'Benign Samples:  {benign_samples}')
                print(f'Attack Samples:  {attack_samples}')
                class_weights = [1. / benign_samples, 1. / attack_samples]
                sample_weights = self.ds_train.data_df.iloc[:,-1]
                sample_weights = sample_weights.apply(lambda x: class_weights[0] if x == 0 else class_weights[1])
                sample_weights = torch.tensor(sample_weights.values)

                self.sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
            return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8, sampler=self.sampler)
        else:
            return DataLoader(self.ds_train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
            return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8)

class SklearnTrainerCallback(Callback):
    def __init__(self, opt, clf_model_name, rae, feature_transformer, s_threshold=[0.001], threshold_name='default'):
        super().__init__()
        self.opt = opt
        self.clf_model_name = clf_model_name
        self.feature_transformer = feature_transformer
        self.rae = rae

        self.s_threshold = s_threshold
        if len(s_threshold) == 1:
            self.s_threshold = s_threshold[0]

        self.threshold_name = threshold_name

        # Create the sklearn model
        self.model_modulename, self.model_classname = clf_model_name.split('.')
        self.model_module = importlib.import_module(f'models.{self.model_modulename}')
        self.model_class = getattr(self.model_module, self.model_classname)

        self.data_loaded = False
        self.training_data = None
        self.validation_data = None

    def recalculate_training_thresholds(self):
        # Recalculate the thresholds based on autoencoder reconstruction
        # Using the full training set of data
        # This mimics the autoencoder's validation_step core functionality

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

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get training and validation data
        if not self.data_loaded:
            if self.opt.use_all_training_data and hasattr(trainer.datamodule.ds_train, 'all_training_data'):
                print('Using ds_train.all_training_data')
                self.training_data = copy.deepcopy(trainer.datamodule.ds_train.all_training_data)
            else:
                self.training_data = copy.deepcopy(trainer.datamodule.ds_train.original_X)

            self.validation_data = copy.deepcopy(trainer.datamodule.ds_val.data_df)
            self.data_loaded = True

            self.recalculate_training_thresholds()

        xtrain, ytrain = self.feature_transformer.transform(trainer, self.rae, self.training_data, threshold=self.s_threshold) 
        xval, yval = self.feature_transformer.transform(trainer, self.rae, self.validation_data, threshold=self.s_threshold)

        # Save off LSO comparison
        if self.feature_transformer.__class__.__name__ == "OriginalLSFeatureTransformer" or self.feature_transformer.__class__.__name__ == "OriginalLSThresholdFeatureTransformer" :
            self.feature_transformer.plot_lso_comparison(self.opt.results_dir, f'val_{self.threshold_name}')

        # Save for reference
        with open(f'{self.opt.results_dir}/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_sample_data.npy', 'wb') as f:
            np.save(f, xval)
            np.save(f, yval)

        # Used on_train_end
        self.ytrain = ytrain
        self.yval = yval

        # Create new model
        if 'knn' in self.model_modulename:
            self.model = self.model_class(n_neighbors=self.opt.n_neighbors)
        elif 'random_forest' in self.model_modulename:
            self.model = self.model_class(max_features=self.opt.rf_max_features)
        elif 'threshold' in self.model_modulename:
            self.model = self.model_class(threshold=self.s_threshold)
        else:
            self.model = self.model_class()

        # Call fit with training data
        self.model.fit(xtrain, ytrain)

        print('VALIDATION:')

        # Perform validation 
        self.predictions = self.model.forward(xval,)
        self.predictions_proba = self.model.predict_proba(xval,)
        self.val_acc = self.model.score(xval, yval)

        # Collect metrics
        self.prec = precision(
            predicted_labels=self.predictions,
            true_labels=yval,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        self.rec = recall(
            predicted_labels=self.predictions,
            true_labels=yval,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        self.f1 = f1_score(
            predicted_labels=self.predictions,
            true_labels=yval,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        # Log our metrics
        pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_precision', self.prec, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_recall', self.rec, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_f1-score', self.f1, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_accuracy', self.val_acc, on_step=False, on_epoch=True)

        if hasattr(self.model.model, 'oob_score_'):
            self.val_oob_score = self.model.model.oob_score_
            pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_oob_score', self.val_oob_score, on_step=False, on_epoch=True)

    def on_train_end(self, trainer, pl_module):
        EXPERIMENTS_DB = './output/experiments_db'

        if 'random_forest' in self.model_modulename:
            importances = self.model.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plot_feature_importance(self.model, self.feature_transformer.num_features, self.feature_transformer.feature_names, metrics_dir=self.opt.results_dir, prefix=f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}')

        # VALIDATION - Get confusion matrix
        cm, cm_norm = plot_confusion_matrix(
            predicted_labels=self.predictions,
            true_labels=self.yval,
            metrics_dir=self.opt.results_dir,
            prefix=f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}',
            is_training=False,
            title=f'{trainer.datamodule.dataset_name} Validation Set',
            target_names=['benign','attack']
        )

        add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'precision', self.prec)
        add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'recall', self.rec)
        add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'f1_score', self.f1)
        add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'accuracy', self.val_acc)

        if hasattr(self.model.model, 'oob_score_'):
            add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'oob_score', self.val_oob_score)

        # Save model
        self.model.feature_names = self.feature_transformer.feature_names
        self.model.save(path=f'{self.opt.results_dir}/{self.opt.save_prefix}_{self.threshold_name}_{self.feature_transformer.__class__.__name__}_{self.model_modulename}_{self.model_classname}-final.pkl')

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save clf model
        self.model.feature_names = self.feature_transformer.feature_names
        self.model.save(path=f'{self.opt.results_dir}/{self.opt.save_prefix}_{self.threshold_name}_{self.feature_transformer.__class__.__name__}_{self.model_modulename}_{self.model_classname}-epoch={pl_module.current_epoch}.pkl')

    def on_test_epoch_start(self, trainer, pl_module):
        # Get training and validation data
        self.test_data = copy.deepcopy(trainer.datamodule.ds_test.data_df)

        # Get transformed features using RAE
        xtest, ytest = self.feature_transformer.transform(trainer, self.rae, self.test_data, threshold=self.s_threshold)

        # Save off LSO comparison
        if self.feature_transformer.__class__.__name__ == "OriginalLSFeatureTransformer" or self.feature_transformer.__class__.__name__ == "OriginalLSThresholdFeatureTransformer":
            self.feature_transformer.plot_lso_comparison(self.opt.results_dir, f'test_{self.threshold_name}')

        # Load RF model
        self.test_predictions = self.model.forward(xtest,)
        self.test_acc = self.model.score(xtest, ytest)

        print('FINAL TEST PREDICTIONS:')

        self.prec = precision(
            predicted_labels=self.test_predictions,
            true_labels=ytest,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        self.rec = recall(
            predicted_labels=self.test_predictions,
            true_labels=ytest,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        self.f1 = f1_score(
            predicted_labels=self.test_predictions,
            true_labels=ytest,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        # Log our metrics
        pl_module.log(f'classifier/test_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_precision', self.prec, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/test_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_recall', self.rec, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/test_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_f1-score', self.f1, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/test_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_accuracy', self.test_acc, on_step=False, on_epoch=True)

        EXPERIMENTS_DB = './output/experiments_db'
        add_metric_to_db(EXPERIMENTS_DB, f'test/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'precision', self.prec)
        add_metric_to_db(EXPERIMENTS_DB, f'test/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'recall', self.rec)
        add_metric_to_db(EXPERIMENTS_DB, f'test/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'f1_score', self.f1)
        add_metric_to_db(EXPERIMENTS_DB, f'test/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'accuracy', self.test_acc)

if __name__ == '__main__':
    # Set up general arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, default='testing', help='Top level project to hold a set of related results.')
    parser.add_argument('--results_dir', required=True, default='./output', help='Directory to store any saved results.')
    parser.add_argument('--experiment', required=True, default='default-experiment', help='Name of the experiment being run.')
    parser.add_argument('--dataset_path', required=True, default=None, help='Dataset to load.')
    parser.add_argument('--save_prefix', required=False, default='', help='Prefix to use when saving the trained model.')
    parser.add_argument('--autoencoder_path', required=False, default=None, help='Path to load autoencoder if needed.')
    parser.add_argument('--feature_type', required=False, default='original', help='Valid options are original|L|S|original_L|original_S|original_LS|LS')
    parser.add_argument('--model', required=False, default='autoencoder_linear.AutoencoderLinear', help='Class name for model to be used')
    parser.add_argument('--clf_model', required=False, default=None, type=str, help='If present, use this classifier.')
    parser.add_argument('--feature_transformer', required=False, default='original_feature_transformer.OriginalFeatureTransformer', type=str, help='Transformer from original features for classifier.')
    parser.add_argument('--batch_size', required=False, default=32, type=int, help='Batch size to be used during training.')
    parser.add_argument('--num_epochs', required=False, default=10000, type=int, help='Number of epochs to perform during training.')
    parser.add_argument('--lr', required=False, default=1, type=float, help='Learning rate to use  during training.')
    parser.add_argument('--list', required=False, default=False, dest='list', action='store_true', help='List models and exit')
    parser.add_argument('--early_stopping_error', required=False, default=1e-7, type=float, help='If the RobustAutoencoder stop criteria reaches this threshold we end training.')
    parser.add_argument('--n_neighbors', required=False, default=10, type=int, help='For KNN classifier, how many neighbors to use.')
    parser.add_argument('--lambda_filter', required=False, default=0.01, type=float, help='Lambda filter for Robust Autoencoder.')
    parser.add_argument('--total_rows_threshold', required=False, default=150000, type=int, help='The total number of rows to use from our data distrubuted among our train/val/test splits.')
    parser.add_argument('--data_module', required=False, default='NetflowDataModule', help='Data module to use for loading data to be used by the model.')
    parser.add_argument('--sklearn', required=False, default=False, dest='sklearn', action='store_true', help='Indicates if classifier is an sklearn model.')
    parser.add_argument('--l2', required=False, default=0.4, type=float, help='Level of regularization to apply to optimzier in classifier model.')
    parser.add_argument('--clf_epochs', required=False, default=100, type=int, help='Number of epochs to perform for training a classifier.')
    parser.add_argument('--rf_max_features', required=False, default='auto', type=str, help='Maximum number of features for each random forest tree to consider.')
    parser.add_argument('--load_from_disk', required=False, default=False, dest='load_from_disk', action='store_true', help='If provided, we load a static dataset from disk, else, we reprocess the data.')
    parser.add_argument('--save_data_prefix', required=False, default='conference', help='Prefix added to any saved data splits to allow for grouping data.')
    parser.add_argument('--reserve_type', required=False, default='rae_lambda_tuning', help='Indicates which saved data split to load [rae_lambda_tuning|rae_classifier_tuning|rae_full_loop_testing]')
    parser.add_argument('--hidden_layer_size', required=False, default=15, type=int, help='Size of the hidden layer for AE.')
    parser.add_argument('--use_all_training_data', required=False, default=False, dest='use_all_training_data', action='store_true', help='Indicates if all_training_data should be used for classifiers.')
    parser.add_argument('--s_threshold', required=False, default='0.01', type=str, help='Comma separated list of threshold values used to filter down S.')
    parser.add_argument('--group', required=False, default='lewbug', help='Group name used to tie runs together in the wandb UI.')

    opt = parser.parse_args()

    print(opt)

    # For now leave this to 0 since we are only using
    # sklearn classifiers and GPUs will run out of memory
    num_gpus = 0

    if opt.list:
        list_models()
        sys.exit(0)

    # We are assuming there is a model in this directory
    # Die hard if the directory doesn't exist
    if not os.path.exists(opt.results_dir):
        print('Could not find mater directory [{opt.results_dir}]')
        sys.exit(1)

    # Get name of dataset
    # globals call is a bit sketchy but it works...
    ds_name = os.path.split(opt.dataset_path)[-1].replace('.pkl', '')
    dm_class = globals()[opt.data_module]
    dm = dm_class(
        data_path=opt.dataset_path, 
        batch_size=opt.batch_size, 
        total_rows_threshold=opt.total_rows_threshold,
        load_from_disk=opt.load_from_disk,
        prefix=opt.save_data_prefix,
        reserve_type=opt.reserve_type,
    )

    # wandb init
    wandb.init(project=opt.project, name=f'{opt.clf_model}_{opt.feature_transformer}_{opt.experiment}-{ds_name}', group=opt.group)

    # Loggers
    wandb_logger = WandbLogger(project=opt.project, name=f'{opt.experiment}-{ds_name}')
    csv_logger = CSVLogger(opt.results_dir, name=f'{opt.experiment}-{ds_name}') 

    model_modulename, model_classname = opt.model.split('.')
    model_module = importlib.import_module(f'models.{model_modulename}')
    model_class = getattr(model_module, model_classname)

    # Load model from disk
    print(f'{opt.results_dir}/last.ckpt')
    ae = model_class.load_from_checkpoint(f'{opt.results_dir}/last.ckpt')
    ae.eval()

    ###
    # CLASSIFIER
    ###

    # Create new NN module
    # - Pass in our trained AE
    # - Fit the NN module and test it 
    clf_models = opt.clf_model.split(',')
    for clf_model_name in clf_models:

        if opt.sklearn:
            clf_model_modulename = 'dummy_model'
            clf_model_classname = 'DummyModel'
            clf_model_module = importlib.import_module(f'models.{clf_model_modulename}')
            clf_model_class = getattr(clf_model_module, clf_model_classname)
        else:
            clf_model_modulename, clf_model_classname = clf_model_name.split('.')
            clf_model_module = importlib.import_module(f'models.{clf_model_modulename}')
            clf_model_class = getattr(clf_model_module, clf_model_classname)

        # Train/validate and test classifier
        classifier_callbacks = []
        classifier_epochs = opt.clf_epochs

        if opt.clf_model and opt.sklearn:
            classifier_epochs = 1
            feature_transformers = opt.feature_transformer.split(',')
            for feature_transformer in feature_transformers:
                transformer_modulename, transformer_classname = feature_transformer.split('.')
                transformer_module = importlib.import_module(f'feature_transformers.{transformer_modulename}')
                transformer_class = getattr(transformer_module, transformer_classname)
                # LEWBUG:  May add this back in later
                ## Have a callback for each threshold we are interested in
                #for threshold in opt.s_threshold.split(','):
                #    sklearn_trainer_callback = SklearnTrainerCallback(opt, clf_model_name, ae, transformer_class(), s_threshold=float(threshold), threshold_name=str(threshold))
                #    classifier_callbacks.append(sklearn_trainer_callback)

                # For our MSE reconstruction classifier, only use the corresponding feature transformer
                # For others, do not use reconstruction transformer
                if 'reconstruction' in clf_model_name:
                    if not 'reconstruction' in transformer_modulename:
                        continue
                if not 'reconstruction' in clf_model_name:
                    if 'reconstruction' in transformer_modulename:
                        continue

                if 'threshold' in transformer_modulename:
                    # Have a callback for automatically calculated thresholds as well
                    for i, threshold in enumerate(ae.threshold_options):
                        if 'feature_threshold' in transformer_modulename:
                            continue

                        sklearn_trainer_callback = SklearnTrainerCallback(opt, clf_model_name, ae, transformer_class(), s_threshold=[threshold], threshold_name=ae.threshold_names[i])
                        classifier_callbacks.append(sklearn_trainer_callback)

                    # Enumerate feature threshold options and only use them with appropriate feature_transformers
                    for i, threshold in enumerate(ae.feature_threshold_options):
                        if 'feature_threshold' not in transformer_modulename:
                            continue

                        sklearn_trainer_callback = SklearnTrainerCallback(opt, clf_model_name, ae, transformer_class(), s_threshold=threshold, threshold_name=ae.feature_threshold_names[i])
                        classifier_callbacks.append(sklearn_trainer_callback)
                else:
                    sklearn_trainer_callback = SklearnTrainerCallback(opt, clf_model_name, ae, transformer_class(), s_threshold=[999], threshold_name='no_threshold')
                    classifier_callbacks.append(sklearn_trainer_callback)


            clf_model = clf_model_class()

            clf_trainer = pl.Trainer(
                gpus=num_gpus, 
                logger=[wandb_logger, csv_logger], 
                max_epochs=classifier_epochs, 
                callbacks=classifier_callbacks,
                progress_bar_refresh_rate=500,
                default_root_dir=opt.results_dir,
            )

            # Adjust data module to use new data for training/validation
            clf_trainer.fit(clf_model, datamodule=dm)
            clf_trainer.test(clf_model, datamodule=dm)

            ###
            # Free up memory
            # - Currently unclear if this part is working
            ###

            print('LEWBUG:  DELETING OBJECTS')
            del clf_trainer
            for callback in classifier_callbacks:
                del callback

        else:

            print('LEWBUG:  In Else Block.')

            # Have a callback for automatically calculated thresholds as well
            for i, threshold in enumerate(ae.threshold_options):
                transformer_modulename, transformer_classname = opt.feature_transformer.split('.')
                transformer_module = importlib.import_module(f'feature_transformers.{transformer_modulename}')
                transformer_class = getattr(transformer_module, transformer_classname)
                feature_transformer = transformer_class()

                if not 'threshold' in transformer_modulename:
                    threshold_name = 'no_threshold'
                else:
                    threshold_name = ae.threshold_names[i]
                    # For unsw, skip anything other than the  25% threshold
                    # to reduce runtime
                    if 'unsw' in opt.dataset_path:
                        if not '25_percent' in threshold_name:
                            continue

                    if 'sf_scenario_6' in opt.dataset_path:
                        if 'high' in threshold_name:
                            continue
                        if 'low' in threshold_name:
                            continue

                    if 'sf_scenario_5' in opt.dataset_path:
                        if not '75_percent' in threshold_name:
                            continue

                print('LEWBUG:', threshold_name)

                clf_checkpoint_callback = ModelCheckpoint(
                    dirpath=f'{opt.results_dir}/clf', 
                    save_top_k=-1, 
                    every_n_val_epochs=25,
                    save_last=True, 
                    filename=f'CLF-{opt.project}-{opt.experiment}-{ds_name}' + '-{epoch:06d}'
                )

                classifier_callbacks = classifier_callbacks + [clf_checkpoint_callback]

                clf_model = clf_model_class(
                    35,
                    3,
                    [128, 128+64, 128+64+64],
                    feature_transformer,
                    threshold_name,
                    opt.results_dir,
                    opt.experiment,
                    1.0,
                    0.5,
                )
                clf_model.rae = ae

                clf_trainer = pl.Trainer(
                    gpus=num_gpus, 
                    logger=[wandb_logger, csv_logger], 
                    max_epochs=classifier_epochs, 
                    callbacks=classifier_callbacks,
                    progress_bar_refresh_rate=500,
                    default_root_dir=opt.results_dir,
                )

                # Adjust data module to use new data for training/validation
                clf_trainer.fit(clf_model, datamodule=dm)
                clf_trainer.test(clf_model, datamodule=dm)

                ###
                # Free up memory
                # - Currently unclear if this part is working
                ###

                print('LEWBUG:  DELETING OBJECTS')
                del clf_trainer
                for callback in classifier_callbacks:
                    del callback

                # Only run once if no threshold involved
                if not 'threshold' in transformer_modulename:
                    break
