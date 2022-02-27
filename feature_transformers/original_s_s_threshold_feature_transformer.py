import torch
import numpy as np
import pandas as pd
import feature_transformers.feature_utilities
import copy

class OriginalSSThresholdFeatureTransformer():
    def __init__(self):
        super().__init__()

    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        L = pl_module.forward(torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device))
        S = torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device) - L
        #S = feature_transformers.feature_utilities.threshold_round_ohe(S, threshold=0.9, round_type='up')
        S_features = copy.deepcopy(S)
        S = feature_transformers.feature_utilities.threshold_mse(S, threshold=threshold)
        #S = feature_transformers.feature_utilities.threshold_round_ohe(S, threshold=threshold, round_type='down')
        #S = feature_transformers.feature_utilities.threshold_round_continuous(S, threshold=threshold, round_type='down')
        original_S_SThresh = np.concatenate((data_df.iloc[:,1:-1], S_features.detach().cpu().numpy(), S.detach().cpu().numpy()), axis=1)

        y = data_df.iloc[:,-1]

        self.num_features = original_S_SThresh.shape[1]
        cols = list(data_df.columns[1:-1])
        cols = cols + [f's_{c}' for c in cols] + [f'st_{c}' for c in cols]
        self.feature_names = np.array(cols)
        return original_S_SThresh, y

    def torch_transform(self, trainer, pl_module, data, label, threshold=0.01):
        L = pl_module.forward(data)
        S = data - L
        S_features = S.detach().clone()
        S = feature_transformers.feature_utilities.threshold_mse(S, threshold=threshold)

        dim = len(S.size()) - 1
        original_S = torch.cat((data, S_features, S), dim=dim)

        self.num_features = original_S.size()[dim]
        self.feature_names = np.array([x for x in range(original_S.size(dim))])
        return original_S, label
