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
from sklearn import preprocessing
import sklearn.model_selection
import sys
import torch.utils.data

class DatasetProcessor():
    '''
    Base class for any processors that will work on a dataset.
    In the future maybe this could also return some information about the change made:
        - Changed columns
        - Renames
        - Drops
    This may have some benefit to inspect how a dataset was altered in preprocessing.
    '''
    def __init__(self):
        pass

    def process(self, df):
        '''
        Perform processing on a pandas dataframe and then returns the updated dataframe.
        This default method just returns the original dataframe.
        '''
        return df

class MakeNiceColumnNames(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        # Strip any leading spaces
        df.columns = df.columns.str.strip()

        # Make columns lower case
        df.columns = [col.lower() for col in df.columns]

        # Replace other contiguous spaces with an underscore
        df.columns = [re.sub('\s+', '_', col) for col in df.columns]

        # Special characters replaced by underscore maybe
        df.columns = [re.sub(r'\.+', '_', col) for col in df.columns]
        df.columns = [re.sub(r'/+', '_', col) for col in df.columns]
        df.columns = [re.sub(r'\\+', '_', col) for col in df.columns]
        df.columns = [re.sub(r':', '_', col) for col in df.columns]
        df.columns = [re.sub(r'-', '_', col) for col in df.columns]

        return df

class MapColumnNames(DatasetProcessor):
    def __init__(self, column_mapping=None):
        super().__init__()
        self.column_mapping=column_mapping

    def process(self, df):
        df.rename(columns=self.column_mapping, inplace=True)
        return df

class DropColumns(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        if self.target_cols:
            df = df.drop(columns=self.target_cols, axis=1)
        return df

class DropIPv6Addresses(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        for col in self.target_cols:
            # IPv6 addresses have colons in their format
            indices_to_drop = df[df[col].str.contains(':')].index
            df.drop(index=indices_to_drop, inplace=True)
        return df

class CleanBadValues(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df

class ReplaceMissingValuesWithMean(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        for col in self.target_cols:
            mean = df[(df[col] != '-') & (df[col] != np.nan)][col].astype(float).mean()
            df[col] = df[col].apply(lambda x: mean if x == '-' or x == np.nan else x)
            df[col] = df[col].astype(float)
        return df

class ReplaceMissingValues(DatasetProcessor):
    def __init__(self, target_cols=None, default_value='default'):
        super().__init__()
        self.target_cols = target_cols
        self.default_value = default_value

    def process(self, df):
        for col in self.target_cols:
            df[col] = df[col].apply(lambda x: self.default_value if x == '-' or x == np.nan else x)
        return df

class StripStringColumns(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        df[self.target_cols] = df[self.target_cols].str.strip()

class OneHotEncodeColumns(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        if self.target_cols:
            df = pd.get_dummies(df, columns=self.target_cols)
        return df

class CleanProtocolColumn(DatasetProcessor):
    def __init__(self, target_col=None):
        super().__init__()
        self.target_col = target_col
        self.protocol_map = {

            # Application Layer Protocols
            'rtp':'rtp',
            'rtcp':'rtp',

            # Transport Layer Protocols
            6:'tcp',
            '6':'tcp',
            'tcp':'tcp',
            17:'udp',
            '17':'udp',
            'udp':'udp',
            'udt':'udp', # UDP Data Transfer Protocol
            'sctp':'sctp', # Stream Control Transmission Protocol

            # Internet Layer Protocols
            'icmp':'icmp',
            'igmp':'igmp',
            'esp':'tunneling', # Encapsulating Security Payload
            'gre':'tunneling', # Generic Routing Encapsulation
            'ipip':'tunneling', # Ip-in-IP Encapsulation
            'ipnip':'tunneling',
            'ip':'ip',
            'ipv6':'ip',
            'ipv6-frag':'ip',
            'ipv6-route':'ip',
            'ipv6-no':'ip', 
            'ipv6-opts':'ip', 

            # Link Layer Protocols
            'arp':'arp',
            'ospf':'ospf',

            # Unknown Protocols
            'other':'other',
            'nan':'other',
            np.NaN:'other',
        }

       # The protocols below are mapped to 'other' for now but do appear

       # 'sun-nd', 'swipe', 'mobile', 'pim', 'ggp',
       # 'st2', 'egp', 'cbt', 'emcon', 'nvp', 'igp', 'xnet', 'argus',
       # 'bbn-rcc', 'chaos', 'pup', 'hmp', 'mux', 'dcn', 'prm', 'trunk-1',
       # 'xns-idp', 'trunk-2', 'leaf-1', 'leaf-2', 'irtp', 'rdp', 'iso-tp4',
       # 'netblt', 'mfe-nsp', 'merit-inp', '3pc', 'xtp', 'idpr', 'tp++',
       # 'ddp', 'idpr-cmtp', 'il', 'idrp', 'sdrp',
       # 'rsvp', 'mhrp', 'bna', 'esp', 'i-nlsp',
       # 'narp', 'tlsp', 'skip', 'any', 'cftp',
       # 'sat-expak', 'kryptolan', 'rvd', 'ippc', 'sat-mon', 'ipcv', 'visa',
       # 'cpnx', 'cphb', 'wsn', 'pvp', 'br-sat-mon', 'wb-mon', 'wb-expak',
       # 'iso-ip', 'secure-vmtp', 'vmtp', 'vines', 'ttp', 'nsfnet-igp',
       # 'dgp', 'tcf', 'eigrp', 'sprite-rpc', 'larp', 'mtp', 'ax.25',
       # 'micp', 'aes-sp3-d', 'encap', 'etherip', 'pri-enc', 'gmtp',
       # 'pnni', 'ifmp', 'aris', 'qnx', 'a/n', 'scps', 'snp', 'ipcomp',
       # 'compaq-peer', 'ipx-n-ip', 'vrrp', 'zero', 'pgm', 'iatp', 'ddx',
       # 'l2tp', 'srp', 'stp', 'smp', 'uti', 'sm', 'ptp', 'fire', 'crtp',
       # 'isis', 'crudp', 'sccopmce', 'sps', 'pipe', 'iplt', 'unas', 'fc',
       # 'ib' 

    def map_unknown_protocols(self, val):
        if val == None:
            return 'other'

        if type(val) == str:
            val = val.lower()
            val = val.strip()

        if type(val) == str and val not in self.protocol_map.keys():
            print(val)
            val = 'other'

        if type(val) == int and val != 6 and val != 17:
            print(val)
            val = 'other'

        return str(val)

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: self.map_unknown_protocols(x))
        df[self.target_col] = df[self.target_col].str.lower().map(self.protocol_map)
        protocol_type = pd.CategoricalDtype(set(self.protocol_map.values()))
        df[self.target_col] = df[self.target_col].astype(protocol_type)
        return df

class DetermineOneDirectionBytes(DatasetProcessor):
    def __init__(self, total_bytes_col='', known_bytes_col='', new_column_name=''):
        super().__init__()
        self.total_bytes_col = total_bytes_col
        self.known_bytes_col = known_bytes_col
        self.new_column_name = new_column_name

    def determine_bytes(self, total_bytes, known_bytes):
        if total_bytes < known_bytes:
            print('ERROR:  total_bytes < known_bytes')
            return 0
        return total_bytes - known_bytes

    def process(self, df):
        df[self.new_column_name] = df.apply(lambda x: self.determine_bytes(x[self.total_bytes_col], x[self.known_bytes_col]), axis=1)
        return df

class EstimatePackets(DatasetProcessor):
    def __init__(self, total_packets_col='', source_bytes_col='', destination_bytes_col=''):
        super().__init__()
        self.total_packets_col = total_packets_col
        self.source_bytes_col = source_bytes_col
        self.destination_bytes_col = destination_bytes_col

    def estimate_source_packets(self, total_packets, source_bytes, destination_bytes):
        if total_packets == 0:
            return 0

        if source_bytes == 0:
            return 0

        # Assume a proportion related to bytes
        # This is probably incorrect in a lot of situations but still
        # reflects total packets for the flow
        total_bytes = source_bytes + destination_bytes
        source_estimate = total_packets * (source_bytes / total_bytes) 

        return source_estimate

    def process(self, df):
        df['source_packets'] = df.apply(lambda x: self.estimate_source_packets(x[self.total_packets_col], x[self.source_bytes_col], x[self.destination_bytes_col]), axis=1)
        df['destination_packets'] = df.apply(lambda x: x[self.total_packets_col] - x['source_packets'], axis=1)
        return df

class SumColumns(DatasetProcessor):
    def __init__(self, new_column_name=None, target_cols=None):
        super().__init__()
        self.new_column_name = new_column_name
        self.target_cols = target_cols

    def process(self, df):
        df[self.new_column_name] = df[self.target_cols].sum(axis=1)
        return df

class AddDefaultColumn(DatasetProcessor):
    def __init__(self, new_column_name=None, default_value=0.0):
        super().__init__()
        self.new_column_name = new_column_name
        self.default_value = default_value

    def process(self, df):
        df[self.new_column_name] = self.default_value
        return df

class MakeLabelColumnLast(DatasetProcessor):
    def __init__(self, target_cols=['label']):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        if self.target_cols:
            df = df[[c for c in df.columns if c not in self.target_cols] + self.target_cols]
        return df

class SortColumnsAlphabetically(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        df = df.reindex(sorted(df.columns), axis=1)
        return df

class MakeTimestampColumnFirst(DatasetProcessor):
    def __init__(self, target_cols=['timestamp']):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        if self.target_cols:
            df = df[self.target_cols + [c for c in df.columns if c not in self.target_cols]]
        return df

class ApplyFunctionToColumns(DatasetProcessor):
    def __init__(self, target_cols=[], function=None):
        super().__init__()
        self.target_cols = target_cols
        self.function = function

    def process(self, df):
        for col in self.target_cols:
            df[col] = df[col].apply(lambda x: self.function(x) if x > 0 else x)
        return df

# Expects standard column names
class AddBytesPerPacketFeatures(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        df['total_bytes_per_packet'] = df.apply(lambda x: x['total_bytes'] / x['total_packets'] if x['total_packets'] != 0 else 0, axis=1)
        df['destination_bytes_per_packet'] = df.apply(lambda x: x['total_destination_bytes'] / x['destination_packets'] if x['destination_packets'] != 0 else 0, axis=1)
        df['source_bytes_per_packet'] = df.apply(lambda x: x['total_source_bytes'] / x['source_packets'] if x['source_packets'] != 0 else 0, axis=1)
        return df

# Expects standard column names
class AddBytesPerSecondFeatures(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        df['total_bytes_per_second'] = df.apply(lambda x: x['total_bytes'] / x['duration'], axis=1)
        df['destination_bytes_per_second'] = df.apply(lambda x: x['total_destination_bytes'] / x['duration'], axis=1)
        df['source_bytes_per_second'] = df.apply(lambda x: x['total_source_bytes'] / x['duration'], axis=1)
        return df

# Expects standard column names
class AddPacketsPerSecondFeatures(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        df['packets_per_second'] = df.apply(lambda x: x['total_packets'] / x['duration'], axis=1)
        df['destination_packets_per_second'] = df.apply(lambda x: x['destination_packets'] / x['duration'], axis=1)
        df['source_packets_per_second'] = df.apply(lambda x: x['source_packets'] / x['duration'], axis=1)
        return df

class IpAddressToByteColumns(DatasetProcessor):
    def __init__(self, prefix='', target_col=None):
        super().__init__()
        self.target_col = target_col
        self.prefix = prefix

    def process(self, df):
        new_cols = [self.prefix + '_byte1', self.prefix + '_byte2', self.prefix + '_byte3', self.prefix + '_byte4']
        df[new_cols] = df[self.target_col].str.split( '.', expand=True)
        return df

class FilterPortsOver1024(DatasetProcessor):
    def __init__(self, target_col):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df.loc[ df[self.target_col] >= 1024, self.target_col] = 1024
        return df

class PrepareAttackBenignLabelCidds(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x.strip() == 'normal' else 1)
        return df

class PrepareAttackBenignLabelNDSec1(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x.strip() == 'NORMAL' else 1)
        return df

class PrepareAttackBenignLabel(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x == 'BENIGN' else 1)
        return df

class PrepareAttackBenignLabelCtu13(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x.rfind('Botnet') != -1 else 0)
        return df

class PrepareAttackBenignLabelIot23(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x == 'Malicious' else 0)
        return df

class PrepareAttackBenignLabelUgr16(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x == 'background' else 1)
        return df

class BinaryEncodeColumns(DatasetProcessor):
    def __init__(self, target_cols=None, new_cols=None, padding=0, wrapper_fn=None):
        super().__init__()
        self.target_cols=target_cols
        self.new_cols=new_cols
        self.padding = padding
        self.wrapper_fn = wrapper_fn

    def get_binary_string(self, bits, padding=0):
        binary_string = bin(int(bits)).replace('0b', '').zfill(padding)
        formatted_string = ''
        for bit in binary_string[:-1]:
            formatted_string += bit + ' ' 
        formatted_string += binary_string[-1]
        return formatted_string

    def process(self, df):
        if len(self.target_cols) != len(self.new_cols):
            print('WARNING:  target_cols and new_cols have different lengths.')
            return df

        for i in range(len(self.target_cols)):
            if self.wrapper_fn is not None:
                df[self.new_cols[i]] = df[self.target_cols[i]].map(lambda x: self.get_binary_string(self.wrapper_fn(x), padding=self.padding))
            else:
                df[self.new_cols[i]] = df[self.target_cols[i]].map(lambda x: self.get_binary_string(x, padding=self.padding))
        return df

class StringToColumns(DatasetProcessor):
    def __init__(self, target_cols=None, prefix=None):
        super().__init__()
        self.target_cols = target_cols
        self.prefix = prefix

    def get_field_names(self, prefix=None, how_many=None):
        if not prefix or not how_many:
            return []
        return [ prefix + str(x) for x in range(how_many) ]
        
    def process(self, df):
        for i in range(len(self.target_cols)):
            sample_col = df[self.target_cols[i]][0].replace(' ', '')
            how_many_columns = len(sample_col)
            cols = self.get_field_names(self.prefix[i], how_many_columns)
            df[cols] = df[self.target_cols[i]].str.split(' ', expand=True)
            df[cols] = df[cols].astype(int)
        return df

class AddNetworkClassColumns(DatasetProcessor):
    def __init__(self, target_cols=None, new_cols=None):
        super().__init__()
        self.target_cols = target_cols
        self.new_cols = new_cols

    def process(self, df):
        if len(self.target_cols) != len(self.new_cols):
            print('WARNING:  Number of target_cols and new_cols does not match.')
            return df

        for i in range(len(self.target_cols)):
            tmp_bytes_df = pd.DataFrame()
            tmp_bytes_df[['byte1', 'byte2', 'byte3', 'byte4']] = df[self.target_cols[i]].str.split('.', expand=True)
            tmp_bytes_df['class'] = tmp_bytes_df['byte1'].apply(
                lambda x: 1 if int(x) <= 127 else 2 if int(x) <= 191 else 3)
            df[self.new_cols[i]] = tmp_bytes_df['class']
        return df

class PerformIpNormalization(DatasetProcessor):
    def __init__(self, target_cols=None, prefix=None):
        super().__init__()
        self.target_cols = target_cols
        self.prefix = prefix

    def process(self, df):
        for i in range(len(self.target_cols)):
            tmp_df = pd.DataFrame()
            tmp_df[['byte1', 'byte2', 'byte3', 'byte4']] = df[self.target_cols[i]].str.split('.', expand=True)
            tmp_df[['byte1', 'byte2', 'byte3', 'byte4']] = \
                    tmp_df[['byte1', 'byte2', 'byte3', 'byte4']].astype(int).astype(CategoricalDtype(np.arange(256)))

            tmp_df[self.prefix[i] + 'network_byte1'] = tmp_df['byte1']

            tmp_df[self.prefix[i] + 'network_byte2'] = tmp_df['byte2']
            tmp_df.loc[tmp_df['byte1'].astype(int) <= 127, self.prefix[i] + 'network_byte2'] = 0

            tmp_df[self.prefix[i] + 'network_byte3'] = tmp_df['byte3']
            tmp_df.loc[tmp_df['byte1'].astype(int) <= 191, self.prefix[i] + 'network_byte3'] = 0

            tmp_df[self.prefix[i] + 'network_byte4'] = 0

            tmp_df[self.prefix[i] + 'host_byte1'] = 0

            tmp_df[self.prefix[i] + 'host_byte2'] = tmp_df['byte2']
            tmp_df.loc[tmp_df['byte1'].astype(int) > 127, self.prefix[i] + 'host_byte2'] = 0

            tmp_df[self.prefix[i] + 'host_byte3'] = tmp_df['byte3']
            tmp_df.loc[tmp_df['byte1'].astype(int) > 191, self.prefix[i] + 'host_byte3'] = 0

            tmp_df[self.prefix[i] + 'host_byte4'] = tmp_df['byte4']

            df[[
                self.prefix[i] + 'network_byte1',
                self.prefix[i] + 'network_byte2',
                self.prefix[i] + 'network_byte3',
                self.prefix[i] + 'network_byte4',
                self.prefix[i] + 'host_byte1',
                self.prefix[i] + 'host_byte2',
                self.prefix[i] + 'host_byte3',
                self.prefix[i] + 'host_byte4']] = tmp_df[[
                    self.prefix[i] + 'network_byte1',
                    self.prefix[i] + 'network_byte2',
                    self.prefix[i] + 'network_byte3',
                    self.prefix[i] + 'network_byte4',
                    self.prefix[i] + 'host_byte1',
                    self.prefix[i] + 'host_byte2',
                    self.prefix[i] + 'host_byte3',
                    self.prefix[i] + 'host_byte4']].astype(int).astype(CategoricalDtype(np.arange(256)))

        return df

class PerformPortNormalization(DatasetProcessor):
    def __init__(self, target_cols=None, prefix=None):
        super().__init__()
        self.target_cols = target_cols
        self.prefix = prefix

    def process(self, df):
        for i in range(len(self.target_cols)):
            col = self.target_cols[i]
            prefix = self.prefix[i]

            df[col] = df[col].apply(lambda x: bin(int(x))[2:].zfill(16))

            df[prefix + 'port_byte1'] = df[col].str[:8]
            df[prefix + 'port_byte1'] = df[prefix + 'port_byte1'].apply(lambda x: int(x, 2))
            df[prefix + 'port_byte1'] = df[prefix + 'port_byte1'].astype(CategoricalDtype(np.arange(256)))

            df[prefix + 'port_byte2'] = df[col].str[8:]
            df[prefix + 'port_byte2'] = df[prefix + 'port_byte2'].apply(lambda x: int(x, 2))
            df[prefix + 'port_byte2'] = df[prefix + 'port_byte2'].astype(CategoricalDtype(np.arange(256)))
        return df

class FilterPortsWellKnownOrNot(DatasetProcessor):
    def __init__(self, target_col=None):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        # Order is important here
        df.loc[ df[self.target_col] < 1024, self.target_col] = 0
        df.loc[ df[self.target_col] >= 1024, self.target_col] = 1
        return df

class FilterColumnsUsingRegex(DatasetProcessor):
    def __init__(self, regex='*'):
        super().__init__()
        self.regex = regex

    def process(self, df):
        df = df.filter(regex=self.regex)
        return df

class OnlyKeepTheseColumns(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols
    
    def process(self, df):
        if self.target_cols:
            df = df[self.target_cols]
        return df

class ReorderAndFilterColumnsUsingSubstring(DatasetProcessor):
    def __init__(self, substrings=None):
        super().__init__()
        self.substrings = substrings 

    def process(self, df):
        col_order = []
        for s in self.substrings:
            col_order = col_order + [col for col in df.columns if s in col]
        df = df[col_order]
        return df

class DropDuplicates(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        df = df.drop_duplicates()
        return df

class IpAddressInternalOrExternal(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        # Not completely accurate as really based on netowrk class but should approximate it for now
        for col in self.target_cols:
            df[col] = df[col].apply(lambda x: 'internal' if int(x.split('.')[0]) >= 192 else 'external')
        return df

class MarkCommsTypeUsingPorts(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols
        self.port_map = {
            0.0:'system_defined',
            21.0:'ftp',
            22.0:'ssh',
            42.0:'dns',
            53.0:'dns',
            135.0:'dns',
            5353.0:'dns',
            80.0:'http',
            443.0:'http',
            88.0:'kerberos',
            123.0:'ntp',
            137.0:'samba',
            138.0:'samba',
            139.0:'samba',
            389.0:'ldap',
            3268.0:'ldap',
            445.0:'ms-smb',
            465.0:'smtps',
            'registered':'registered',
            'user_application':'user_application',
            'nan':'other',
            np.NaN:'other',
        }

    def map_general_ports(self, val):
        if type(val) == str and val.rfind('0x') != -1:
            val = int(val, base=16)
        else:
            val = int(val)

        if val in self.port_map.keys():
            return val 

        if val >= 1024 and val <= 49151:
            return 'registered'

        if val >= 49152:
            return 'user_application'

        # If none of our cases are met, just return the same value
        return val

    def process(self, df):
        for col in self.target_cols:
            df[col] = df[col].apply(lambda x: self.map_general_ports(x))
            df[col] = df[col].map(self.port_map)
            df[col] = df[col].astype(str)
        return df

class MarkCommsTypeUsingBothPorts(DatasetProcessor):
    def __init__(self, new_column_name='communication_type', port1=None, port2=None):
        super().__init__()
        self.port1 = port1
        self.port2 = port2
        self.new_column_name = new_column_name
        self.port_map = {
            21.0:'ftp',
            22.0:'ssh',
            42.0:'dns',
            53.0:'dns',
            135.0:'dns',
            5353.0:'dns',
            80.0:'http',
            443.0:'http',
            88.0:'kerberos',
            123.0:'ntp',
            137.0:'samba',
            138.0:'samba',
            139.0:'samba',
            389.0:'ldap',
            3268.0:'ldap',
            445.0:'ms-smb',
            465.0:'smtps',
            'nan':'other',
            np.NaN:'other',
            'other':'other',
            '-':'other',
        }

    def map_ports(self, port1, port2):
        # These two checks allow us to use either port to 
        # indicate the communication type for a given flow
        # If not in our map, we really don't care about it
        if port1 in self.port_map.keys():
            return port1

        if port2 in self.port_map.keys():
            return port2

        if type(port1) == str and port1.rfind('0x') != -1:
            port1 = int(port1, base=16)
        else:
            port1 = int(port1)

        if type(port2) == str and port2.rfind('0x') != -1:
            port2 = int(port2, base=16)
        else:
            port2 = int(port2)

        # These two checks allow us to use either port to 
        # indicate the communication type for a given flow
        # If not in our map, we really don't care about it
        if port1 in self.port_map.keys():
            return port1

        if port2 in self.port_map.keys():
            return port2

        # If none of our cases are met, just return the same value
        return 'other'

    def process(self, df):
        df[self.new_column_name] = df.apply(lambda x: self.map_ports(x[self.port1], x[self.port2]), axis=1)
        df[self.new_column_name] = df[self.new_column_name].map(self.port_map)
        comms_type = pd.CategoricalDtype(set(self.port_map.values()))
        df[self.new_column_name] = df[self.new_column_name].astype(comms_type)
        return df

class ConvertDurationFromMicrosecondsToSeconds(DatasetProcessor):
    def __init__(self, target_col='duration'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col] * 1e-6
        return df

class FixZeroDurations(DatasetProcessor):
    def __init__(self, target_column='duration', new_value=1e-6):
        super().__init__()
        self.target_col = target_column
        self.new_value = new_value

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: x if x > 0 else 1e-6)
        return df

class SortByDateColumn(DatasetProcessor):
    def __init__(self, target_col=None):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = pd.to_datetime(df[self.target_col])
        df = df.sort_values(by=self.target_col)
        return df

class AddTimeOfDayToDateColumn(DatasetProcessor):
    def __init__(self, target_col=None):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(
            lambda x:  x + ' PM' 
            if ' 12:' in x or
                ' 01:' in x or ' 1:' in x or
                ' 02:' in x or ' 2:' in x or
                ' 03:' in x or ' 3:' in x or
                ' 04:' in x or  ' 4:'in x or
                ' 05:' in x or ' 5:' in x
            else x + ' AM'
            )
        return df


class ConvertStringBytesToNumericBytes(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def convert_to_float(self, val):
        if type(val) == int or type(val) == float:
            return float(val)

        if 'K' in val:
            val = val.replace('M', '')
            val = val.strip()
            val = float(val) * 1e3
            return val


        if 'M' in val:
            val = val.replace('M', '')
            val = val.strip()
            val = float(val) * 1e6
            return val

        if 'G' in val:
            val = val.replace('G', '')
            val = val.strip()
            val = float(val) * 1e9
            return val

        if type(val) == str:
            val = val.strip()
            return float(val)

    def process(self, df):
        for col in self.target_cols:
            df[col] = df[col].apply(lambda x: self.convert_to_float(x))
        return df

class FormatDateTime(DatasetProcessor):
    def __init__(self, target_col=None, original_format=None, target_format=None):
        super().__init__()
        self.target_col = target_col
        self.original_format = original_format
        self.target_format = target_format

    def process(self, df):
        df[self.target_col] = pd.to_datetime(df[self.target_col], format=self.original_format)
        return df

class BalanceUsingMinorityClass(DatasetProcessor):
    def __init__(self, target_col='label', majority_multiplier=1):
        super().__init__()
        self.taret_col = target_col
        self.majority_multiplier = majority_multiplier

    def process(self, df):
        total_samples = df.shape[0]
        num_attack_samples = df.label.sum()
        num_benign_samples = total_samples - num_attack_samples

        attack_samples = df.loc[df.label == 1]
        benign_samples = df.loc[df.label == 0]

        if num_attack_samples <= num_benign_samples:
            base_sample_amount = num_attack_samples
            attack_sample_amount = base_sample_amount

            if base_sample_amount * self.majority_multiplier < num_benign_samples:
                benign_sample_amount = base_sample_amount * self.majority_multiplier
            else:
                benign_sample_amount = num_benign_samples
        else:
            base_sample_amount = num_benign_samples
            benign_sample_amount = base_sample_amount

            if base_sample_amount * self.majority_multiplier < num_attack_samples:
                attack_sample_amount = base_sample_amount * self.majority_multiplier
            else:
                attack_sample_amount = num_attack_samples

        new_df = pd.concat([attack_samples.sample(n=attack_sample_amount),
                            benign_samples.sample(n=benign_sample_amount)])

        # Just to shuffle
        df = copy.deepcopy(new_df.sample(frac=1))

        return df

class SampleWithStratify(DatasetProcessor):
    def __init__(self, num_samples=50000):
        super().__init__()
        self.num_samples = num_samples

    def process(self, df):
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(int(np.rint(self.num_samples *  len(x)/ len(df))))).sample(frac=1).reset_index(drop=True)
        return df

class DropRowsWithNegativeValues(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        df = df[(df[self.target_cols] >= 0).all(1)]
        return df
