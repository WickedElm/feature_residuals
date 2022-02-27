#!/usr/bin/env python

import argparse
import copy
import inquirer
import ipaddress
import math
import numpy as np
import os
import pandas as pd
from pandas.api.types import CategoricalDtype
import ipdb
import pickle
import re
from scipy.io import arff
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.model_selection
import sys
import torch.utils.data
from dataset import DatasetProcessors

###
# What is a dataset?
# - Contains a base source of data
# - Put into a standard format to enable processing
# - A number of transformations performed on the dataset
# - Able to tell us how many fields it has (with and without the label)
###

class Dataset():
    def __init__(self, name='dataset', dataset_source=None, encoding='ISO-8859-1', sep=',', init_processors=None, processors=[], cols_to_drop=[], cols_to_onehot=[], cols_to_normalize=[]):
        pd.set_option('mode.chained_assignment', 'raise')

        self.name = name
        self.init_processors = init_processors
        self.processors = processors
        self.cols_to_drop = cols_to_drop
        self.cols_to_onehot = cols_to_onehot
        self.cols_to_normalize = cols_to_normalize
        self.normalized_columns = []
        self.dataset_source = dataset_source
        self.original_data = self.load_source_data(self.dataset_source, encoding=encoding, sep=sep)
        self.training_means = None
        self.training_stds = None
        self.scaler = None

        # Is this a reference or do I need a deep copy here?
        # For current usage probably doesn't matter
        self.processed_data = copy.deepcopy(self.original_data)

        # Only generated if the client calls the split method
        self.training_data = pd.DataFrame()
        self.validation_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

        self.training_indices = pd.core.indexes.numeric.Int64Index([])
        self.validation_indices = pd.core.indexes.numeric.Int64Index([])
        self.test_indices = pd.core.indexes.numeric.Int64Index([])

        self.benign_reserve = None

    def save_indices(self):
        if not self.training_data.empty:
            self.training_indices = self.training_data.index

        if not self.validation_data.empty:
            self.validation_indices = self.validation_data.index

        if not self.test_data.empty:
            self.test_indices = self.test_data.index

    def write_indices_to_disk(self, save_path, save_prefix=''):
        self.write_index_to_disk(f'{save_path}/{save_prefix}_training_indices.pkl', self.training_indices)
        self.write_index_to_disk(f'{save_path}/{save_prefix}_validation_indices.pkl', self.validation_indices)
        self.write_index_to_disk(f'{save_path}/{save_prefix}_test_indices.pkl', self.test_indices)

    def write_index_to_disk(self, save_path, index):
        with open(save_path, 'wb') as f:
            pickle.dump(index, f)

    def sample_data_with_indices(self, training_path, validation_path, test_path):
        with open(training_path, 'rb') as f:
            self.training_indices = pickle.load(f)

        with open(validation_path, 'rb') as f:
            self.validation_indices = pickle.load(f)

        with open(test_path, 'rb') as f:
            self.test_indices = pickle.load(f)

        self.training_data = copy.deepcopy(self.processed_data.loc[self.training_indices])
        self.validation_data = copy.deepcopy(self.processed_data.loc[self.validation_indices])
        self.test_data = copy.deepcopy(self.processed_data.loc[self.test_indices])

    def load_source_data(self, dataset_source, encoding='ISO-8859-1', sep=','):
        '''
        dataset_source:  The path to a valid dataset to load.
                         Expected to be a csv file for now.
        Loads the data found in file [dataset_source] into a Pandas DF.
        '''
        if dataset_source.endswith('.arff'):
            tmp_data = arff.loadarff(dataset_source)
            df = pd.DataFrame(tmp_data[0])
        else:
            df = pd.read_csv(dataset_source, encoding=encoding, sep=sep, skip_blank_lines=True)
        return df

    def add_source_data(self, dataset_source, encoding='ISO-8859-1', sep=','):
        new_df = self.load_source_data(dataset_source)
        self.original_data = pd.concat([self.original_data, new_df], ignore_index=True)
        self.processed_data = copy.deepcopy(self.original_data)

    def perform_init_processing(self):
        for processor in self.init_processors:
            self.processed_data = processor.process(self.processed_data)

        # Make the label column last
        self.processed_data = DatasetProcessors.MakeLabelColumnLast().process(self.processed_data)
        return self.processed_data

    def save(self, save_path, processors=[]):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        processors_path, _ = os.path.split(save_path)
        processors_file_path = f'{processors_path}/processors.pkl'
        with open(processors_file_path, 'wb') as f:
            pickle.dump(processors, f, protocol=pickle.HIGHEST_PROTOCOL)

    def perform_processing(self):
        '''
        Takes the original datasource file and generates processed version
        using the processors contained in the dataset.
        Processors are handled in their original order.
        '''
        for processor in self.processors:
            self.processed_data = processor.process(self.processed_data)

        # Make the label column last
        self.processed_data = DatasetProcessors.MakeLabelColumnLast().process(self.processed_data)
        return self.processed_data

    def reserve_benign_data(self, total_rows_threshold=600000, reserve_rows=100000, reserve_percentage=0.2):
        total_rows = self.processed_data.shape[0]
        if total_rows >= total_rows_threshold:
            self.benign_reserve = self.processed_data[self.processed_data.label == 0].sample(reserve_rows)
            self.processed_data = self.processed_data.drop(self.benign_reserve.index)
        else:
            self.benign_reserve = self.processed_data[self.processed_data.label == 0].sample(frac=reserve_percentage)
            self.processed_data = self.processed_data.drop(self.benign_reserve.index)

    def sample_data(self, total_rows_threshold=500000):
        total_rows = self.processed_data.shape[0]
        if total_rows > total_rows_threshold:
            processed_data, _, _, _ = sklearn.model_selection.train_test_split(
                self.processed_data, 
                self.processed_data.iloc[:,-1],
                train_size=total_rows_threshold,
                stratify=self.processed_data.iloc[:,-1]
            )
            self.processed_data = copy.deepcopy(processed_data)

    def append_processor(self, processor=None, process=True):
        '''
        Appends a processor to the end of the processing list
        processor:  The processor to append.
        process:  True if the processor should be applied.
                  After its application the label column remains last.
        '''
        if processor:
            self.processors.append(processor)

            if process:
                if not self.processed_data.empty:
                    self.processed_data = processor.process(self.processed_data)
                    self.processed_data = DatasetProcessors.MakeLabelColumnLast().process(self.processed_data)

                if not self.test_data.empty:
                    self.test_data = processor.process(self.test_data)
                    self.test_data = DatasetProcessors.MakeLabelColumnLast().process(self.test_data)

                if not self.validation_data.empty:
                    self.validation_data = processor.process(self.validation_data)
                    self.validation_data = DatasetProcessors.MakeLabelColumnLast().process(self.validation_data)

                if not self.training_data.empty:
                    self.training_data = processor.process(self.training_data)
                    self.training_data = DatasetProcessors.MakeLabelColumnLast().process(self.training_data)

        return self.processed_data

    def print_dataset_info(self):
        '''
        Returns the following statistics regarding the dataset object:
        - Name
        - dataset_source
        - Processed number of features
        '''
        print('Dataset Name:  ' + self.name)
        print('Original Source:  ' + self.dataset_source)

        print('Processors Applied:')
        for p in  self.processors:
            print(p)

        print('Number of Features:  ' + str(len(self.processed_data.columns) - 1))
        print('Columns: ')
        print('\n'.join(self.processed_data.columns))
        print(f'Number Training Rows:  {self.training_data.shape[0]}')
        print(f'Number Validation Rows:  {self.validation_data.shape[0]}')
        print(f'Number Test Rows:  {self.test_data.shape[0]}')

    def number_of_features(self):
        return len(self.processed_data.columns) - 1

    def number_of_rows(self):
        return len(self.processed_data)

    def split(self, split_point_1=0.15, split_point_2=0.3, shuffle=True):
        if shuffle:
            self.processed_data = self.processed_data.sample(frac=1)

        if split_point_2 == 0.0:
            splits = [int(split_point_1 * self.number_of_rows())]
            self.test_data , self.training_data = np.split(self.processed_data, splits)
            self.validation_data = self.test_data
        else:
            splits = [int(split_point_1 * self.number_of_rows()), int(split_point_2 * self.number_of_rows())]
            self.test_data , self.validation_data, self.training_data = np.split(self.processed_data, splits)

        return self.test_data, self.validation_data, self.training_data

    def load_data_split_from_disk(self, load_data_path, prefix, reserve_type):
        training_data_path = f'{load_data_path}/{prefix}_{self.name}_{reserve_type}_train.pkl'
        validation_data_path = f'{load_data_path}/{prefix}_{self.name}_{reserve_type}_validation.pkl'
        test_data_path = f'{load_data_path}/{prefix}_{self.name}_{reserve_type}_test.pkl'

        if not os.path.exists(training_data_path):
            return False

        with open(training_data_path, 'rb') as f:
            training_data = pickle.load(f)

        print(f'Loaded {training_data_path}')

        with open(validation_data_path, 'rb') as f:
            validation_data = pickle.load(f)

        print(f'Loaded {validation_data_path}')

        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)

        print(f'Loaded {test_data_path}')

        self.training_data = copy.deepcopy(training_data)
        self.validation_data = copy.deepcopy(validation_data)
        self.test_data = copy.deepcopy(test_data)

        return True

    def reserve_test_set(self, save_path, split_point, shuffle=True, force=False):
        # If reserve exists then just load it and remove from processing
        # Otherwise create it using split_point
        if os.path.exists(save_path) and not force:
            with open(save_path, 'rb') as f:
                test_data = pickle.load(f)
            self.processed_data.drop(test_data.index, inplace=True, errors='ignore')
        else:
            if shuffle:
                self.processed_data = self.processed_data.sample(frac=1)

            _, test_data, _, y_test = sklearn.model_selection.train_test_split(
                self.processed_data, 
                self.processed_data.iloc[:,-1], 
                test_size=split_point, 
                stratify=self.processed_data.iloc[:,-1]
            )

            # Save indices and remove them from processed data
            with open(save_path, 'wb') as f:
                pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.processed_data.drop(test_data.index, inplace=True)

        test_data.columns = self.processed_data.columns
        self.test_data = copy.deepcopy(test_data)

    def save_split_to_disk(self, load_data_path, prefix, reserve_type, main_split_point, split_point_1, split_point_2):
        training_data_path = f'{load_data_path}/{prefix}_{self.name}_{reserve_type}_train.pkl'
        validation_data_path = f'{load_data_path}/{prefix}_{self.name}_{reserve_type}_validation.pkl'
        test_data_path = f'{load_data_path}/{prefix}_{self.name}_{reserve_type}_test.pkl'

        # Shuffle
        self.processed_data = self.processed_data.sample(frac=1)

        # Sample using main split point
        if main_split_point == 1.0:
            sample_to_use = copy.deepcopy(self.processed_data)
            remaining_data = copy.deepcopy(self.processed_data)
        else:
            sample_to_use, remaining_data, y_sample_to_use, y_remaining = sklearn.model_selection.train_test_split(
                self.processed_data, 
                self.processed_data.iloc[:,-1], 
                test_size=main_split_point, 
                stratify=self.processed_data.iloc[:,-1]
            )

        # Remove those samples from processed data
        self.processed_data = copy.deepcopy(remaining_data)

        # Perform  stratified split to get train/val/test
        training_data, test_val_data, y_train, y_test_val = sklearn.model_selection.train_test_split(
            sample_to_use, 
            sample_to_use.iloc[:,-1],
            test_size=split_point_2,
            stratify=sample_to_use.iloc[:,-1]
        )

        # Determine how to split the remaining data between validation and test
        new_split_point = split_point_1 / split_point_2
        validation_data, test_data, y_validation, y_test = sklearn.model_selection.train_test_split(
            test_val_data, 
            test_val_data.iloc[:,-1], 
            test_size=new_split_point,
            stratify=test_val_data.iloc[:,-1]
        )

        self.training_data = copy.deepcopy(training_data)
        self.validation_data = copy.deepcopy(validation_data)
        self.test_data = copy.deepcopy(test_data)

        # Save splits to disk
        with open(training_data_path, 'wb') as f:
            pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(validation_data_path, 'wb') as f:
            pickle.dump(validation_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(test_data_path, 'wb') as f:
            pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def default_stratified_split(self, split_point_1=0.15, split_point_2=0.3, shuffle=True, balance_training=False, majority_multiplier=1):
        if shuffle:
            self.processed_data = self.processed_data.sample(frac=1)

        # Single training / test split if split_point_2 is 0.0
        if split_point_2 == 0.0:
            training_data, validation_data, y_train, y_validation = sklearn.model_selection.train_test_split(
                self.processed_data, 
                self.processed_data.iloc[:,-1], 
                test_size=split_point_1, 
                stratify=self.processed_data.iloc[:,-1]
            )
            test_data = validation_data
        else:
            training_data, test_val_data, y_train, y_test_val = sklearn.model_selection.train_test_split(
                self.processed_data, 
                self.processed_data.iloc[:,-1],
                test_size=split_point_2,
                stratify=self.processed_data.iloc[:,-1]
            )

            # Determine how to split the remaining data between validation and test
            new_split_point = split_point_1 / split_point_2
            validation_data, test_data, y_validation, y_test = sklearn.model_selection.train_test_split(
                test_val_data, 
                test_val_data.iloc[:,-1], 
                test_size=new_split_point,
                stratify=test_val_data.iloc[:,-1]
            )

        if balance_training:
            training_data = DatasetProcessors.BalanceUsingMinorityClass(majority_multiplier=majority_multiplier).process(training_data)

        self.training_data = copy.deepcopy(training_data)
        self.validation_data = copy.deepcopy(validation_data)
        if self.test_data.empty:
            self.test_data = copy.deepcopy(test_data)

        # Mainly used when we have a reserve set but
        # Use training for AE, validation and this set will 
        # used for the NN clf training/validation
        self.extra_test_data = copy.deepcopy(test_data)

        return self.test_data, self.validation_data, self.training_data

    def filter_by_dates(self, target_col, interval_start, interval_end):
        self.processed_data = self.processed_data.loc[
            ((self.processed_data[target_col] >= interval_start) & 
            (self.processed_data[target_col] <= interval_end)),:
        ].copy()

    # Expects 3 date intervals; Assumes 23 hour clock
    def split_by_dates(self, target_col, training_interval, validation_interval, testing_interval):
        self.training_data = self.processed_data.loc[
            ((self.processed_data[target_col] >= training_interval[0]) & 
            (self.processed_data[target_col] <= training_interval[1])),:
        ].copy()

        self.validation_data = self.processed_data.loc[
            ((self.processed_data[target_col] >= validation_interval[0]) & 
            (self.processed_data[target_col] <= validation_interval[1])),:
        ].copy()

        self.test_data = self.processed_data.loc[
            ((self.processed_data[target_col] >= testing_interval[0]) & 
            (self.processed_data[target_col] <= testing_interval[1])),:
        ].copy()

        return self.test_data, self.validation_data, self.training_data

    def split_by_stratified_shuffle(self, n_splits=2, test_size=0.2):
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        sss.get_n_splits(
            self.processed_data.iloc[:, :-1],
            self.processed_data.iloc[:,-1]
        )

        split = sss.split(
            self.processed_data.iloc[:, :-1],
            self.processed_data.iloc[:,-1]
        )

        for train_index, test_index in split:
            self.training_data = self.processed_data.iloc[train_index,:].copy()
            self.validation_data = self.processed_data.iloc[test_index,:].copy()
            self.test_data = self.validation_data.copy()

    def normalize_columns(self, target_cols=None, scaler=None):
        self.normalized_columns = target_cols
        if self.training_data.empty:
            print('WARNING:  No training data.  Skipped normalization')
            return self.test_data, self.validation_data, self.training_data

        # Do not create a new scaler if one was passed in
        if not scaler:
            scaler = sklearn.preprocessing.StandardScaler().fit(self.training_data[target_cols])

        # Store the scaler incase we have
        # disjoint datasets (two different dataset objects) with
        # one used as training and the other as test/validation
        # In this case, we should use the scaler from the training
        # data for normalization
        self.scaler = scaler

        self.training_data[target_cols] = self.scaler.transform(self.training_data[target_cols])
        self.validation_data[target_cols] = self.scaler.transform(self.validation_data[target_cols])
        self.test_data[target_cols] = self.scaler.transform(self.test_data[target_cols])

        return self.test_data, self.validation_data, self.training_data

    def normalize_columns_min_max(self, target_cols=None, scaler=None):
        self.normalized_columns = target_cols
        if self.training_data.empty:
            print('WARNING:  No training data.  Skipped normalization')
            return self.test_data, self.validation_data, self.training_data

        # Do not create a new scaler if one was passed in
        if not scaler:
            scaler = sklearn.preprocessing.MinMaxScaler().fit(self.training_data[target_cols])

        # Store the scaler incase we have
        # disjoint datasets (two different dataset objects) with
        # one used as training and the other as test/validation
        # In this case, we should use the scaler from the training
        # data for normalization
        self.scaler = scaler

        self.training_data[target_cols] = self.scaler.transform(self.training_data[target_cols])
        self.validation_data[target_cols] = self.scaler.transform(self.validation_data[target_cols])
        self.test_data[target_cols] = self.scaler.transform(self.test_data[target_cols])

        return self.test_data, self.validation_data, self.training_data

    # We know for cyber data the minimum possible value is 0 for all of our
    # features.  So we use  0 as the min which prevents any negative values
    # for new data that we normalize.
    def normalize_columns_zero_max(self, target_cols=None, scaler=None):
        self.normalized_columns = target_cols
        if self.training_data.empty:
            print('WARNING:  No training data.  Skipped normalization')
            return self.test_data, self.validation_data, self.training_data

        # Do not create a new scaler if one was passed in
        if not scaler:
            scaler = self.training_data[target_cols].max(axis=0)
            scaler = scaler.apply(lambda x: 1e-9 if x == 0 else x)
            
        # Store the scaler incase we have
        # disjoint datasets (two different dataset objects) with
        # one used as training and the other as test/validation
        # In this case, we should use the scaler from the training
        # data for normalization
        self.scaler = scaler

        self.training_data[target_cols] = self.training_data[target_cols] / self.scaler
        self.validation_data[target_cols] = self.validation_data[target_cols] / self.scaler
        self.test_data[target_cols] = self.test_data[target_cols] / self.scaler

        return self.test_data, self.validation_data, self.training_data

    def _znorm(self, df, means, stds):
        # If all values are zero we will get a divide by zero
        # So we just set it to 0.0 in this case
        df = (df - means) / stds
        df.replace(np.NaN, 0.0, inplace=True)
        return df

    def to_csv(self, output_dir='.'):
        os.makedirs(output_dir, exist_ok=True)

        if not self.test_data.empty:
            self.test_data.to_csv(output_dir + '/test_' + self.name + '.csv', index=False)

        if not self.validation_data.empty:
            self.validation_data.to_csv(output_dir + '/validation_' + self.name + '.csv', index=False)

        if not self.training_data.empty:
            self.training_data.to_csv(output_dir + '/train_' + self.name + '.csv', index=False)

        if self.test_data.empty and self.validation_data.empty and self.training_data.empty:
            if not self.processed_data.empty:
                self.processed_data.to_csv(output_dir + '/' + self.name + '.csv', index=False)

    def reprocess(self):
        # Clear out existing data except for original
        self.processed_data = copy.deepcopy(self.original_data)
        self.training_data = pd.DataFrame()
        self.validation_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

        # Process the data again
        self.perform_processing()

    def reindex(self):
        self.training_data.index = np.arange(len(self.training_data))
        self.validation_data.index = np.arange(len(self.validation_data))
        self.test_data.index = np.arange(len(self.test_data))

    @staticmethod
    def load(dataset_path=None):
        if dataset_path:
            if os.path.exists(dataset_path):
                # Load dataset 
                with open(dataset_path, 'rb') as f:
                    ds = pickle.load(f)
                ds.reindex()

            # Load processors if present
            processors_path, _  = os.path.split(dataset_path)
            processors_file_path = f'{processors_path}/processors.pkl' 
            if os.path.exists(processors_file_path):
                with open(processors_file_path, 'rb') as f:
                    processors = pickle.load(f)
                ds.processors = processors

            return ds
