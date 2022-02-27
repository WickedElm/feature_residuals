import torch
import numpy as np
import pandas as pd
import ipdb

def get_mse(data):
    data_ones = torch.ones(data.shape[0])

    # Get mse of each row 
    data_squared = data * data
    data_means = data_squared.mean(axis=1)
    all_means = (data_ones.T * data_means).T
    return all_means

def get_squared_error(data):
    data_ones = torch.ones(data.shape[0])

    # Get mse of each row 
    data_squared = data * data
    data_sums = data_squared.sum(axis=1)
    all_sums = (data_ones.T * data_sums).T
    return all_sums 

def get_abs_error(data):
    data_ones = torch.ones(data.shape[0])

    # Get mse of each row 
    data_abs = torch.abs(data)
    data_sum = data_abs.sum(axis=1)
    all_sums = (data_ones.T * data_sum).T
    return all_sums

# Same as get_mse but returns means in shape of 
# original input data to be used in torch.where statements
def get_expanded_mse(data):
    data_ones = torch.ones_like(data)

    # Get mse of each row 
    data_squared = data * data

    if len(data.size()) == 3:
        data_means = data_squared.mean(axis=2)
        all_means = torch.ones_like(data)
        for i in range(data.size()[0]):
            all_means[i,:,:] = (data_ones[i,:,:].T * data_means[i]).T
    else:
        data_means = data_squared.mean(axis=1)
        all_means = (data_ones.T * data_means).T

    return all_means

def threshold_mse(data, threshold=0.65, round_type='down', device=torch.device('cpu')):
    if torch.is_tensor(threshold):
        threshold = threshold.float().to(device)

    data_zero = torch.zeros_like(data)

    all_means = get_expanded_mse(data)

    # Apply threshold
    output_data = torch.where(all_means <= threshold, data_zero, data)

    return output_data

def threshold_mse_features(data, threshold=[0.65], round_type='down', device=torch.device('cpu')):
    output_data = torch.zeros_like(data)

    for i in range(data.size()[1]):
        data_zero = torch.zeros_like(data[:,i])
        current_threshold = threshold[i]

        # Apply threshold
        if torch.is_tensor(current_threshold):
            current_threshold = current_threshold.float().to(device)

        output_data[:,i] = torch.where(data[:,i] <= current_threshold, data_zero, data[:,i])

    return output_data

def threshold_absolute_vs_mse(errors, threshold=0.65, round_type='down', device=torch.device('cpu')):
    if torch.is_tensor(threshold):
        threshold = threshold.float().to(device)

    data_zero = torch.zeros_like(errors)
    data_ones = torch.ones_like(errors)

    abs_errors = torch.abs(errors)
    total_row_errors = abs_errors.sum(axis=1)

    # Put into format usable with torch.where
    all_total_row_errors = (data_ones.T * total_row_errors).T

    # Apply threshold
    output_data = torch.where(all_total_row_errors <= threshold, data_zero, errors)

    return output_data

def threshold_squared_error(errors, threshold=0.65, round_type='down', device=torch.device('cpu')):
    if torch.is_tensor(threshold):
        threshold = threshold.float().to(device)

    data_zero = torch.zeros_like(errors)
    data_ones = torch.ones_like(errors)

    squared_errors = errors * errors

    total_row_squared_errors = squared_errors.sum(axis=1)

    # Put into format usable with torch.where
    all_total_squared_errors = (data_ones.T * total_row_squared_errors).T

    # Apply threshold
    output_data = torch.where(all_total_squared_errors <= threshold, data_zero, errors)

    return output_data

def threshold_round_ohe(data, threshold=0.65, round_type='up', device=torch.device('cpu')):
    if torch.is_tensor(threshold):
        threshold = threshold.float().to(device)

    comms_type_ones = torch.ones_like(data[:,:11])
    comms_type_zeros = torch.zeros_like(data[:,:11])
    protocol_ones = torch.ones_like(data[:,17:27])
    protocol_zeros = torch.zeros_like(data[:,17:27])

    if round_type == 'up':
        data[:,:11] = torch.where(data[:,:11] > threshold, comms_type_ones, data[:,:11])
        data[:,17:27] = torch.where(data[:,17:27] > threshold, protocol_ones, data[:,17:27])
    else:
        data[:,:11] = torch.where(data[:,:11] < threshold, comms_type_zeros, data[:,:11])
        data[:,17:27] = torch.where(data[:,17:27] < threshold, protocol_zeros, data[:,17:27])

    return data

def threshold_round_continuous(data, threshold=0.1, round_type='down', device=torch.device('cpu')):
    if torch.is_tensor(threshold):
        threshold = threshold.float().to(device)

    c1_ones = torch.ones_like(data[:,11:17])
    c1_zeros = torch.zeros_like(data[:,11:17])
    c2_ones = torch.ones_like(data[:,27:])
    c2_zeros = torch.zeros_like(data[:,27:])

    if round_type == 'down':
        data[:,11:17] = torch.where(data[:,11:17] < threshold, c1_zeros, data[:,11:17])
        data[:,27:] = torch.where(data[:,27:] < threshold, c2_zeros, data[:,27:])
    else:
        data[:,11:17] = torch.where(data[:,11:17] > threshold, c1_ones, data[:,11:17])
        data[:,27:] = torch.where(data[:,27:] > threshold, c2_ones, data[:,27:])

    return data
