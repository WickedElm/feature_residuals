import torch
import numpy as np
import pandas as pd
import feature_transformers.feature_utilities
import copy

class SSThresholdFeatureTransformer():
    def __init__(self):
        super().__init__()

    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        L = pl_module.forward(torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device))
        S = torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device) - L
        #S = feature_transformers.feature_utilities.threshold_round_ohe(S, threshold=0.9, round_type='up')
        SThresh = copy.deepcopy(S)
        SThresh = feature_transformers.feature_utilities.threshold_mse(SThresh, threshold=threshold)
        #SThresh = feature_transformers.feature_utilities.threshold_round_ohe(SThresh, threshold=threshold, round_type='down')
        #SThresh = feature_transformers.feature_utilities.threshold_round_continuous(SThresh, threshold=threshold, round_type='down')
        SSThresh = np.concatenate((S.detach().cpu().numpy(), SThresh.detach().cpu().numpy()), axis=1)

        y = data_df.iloc[:,-1]

        self.num_features = SSThresh.shape[1]
        original_cols = list(data_df.columns[1:-1])
        cols = [f's_{c}' for c in original_cols] + [f'st_{c}' for c in original_cols]
        self.feature_names = np.array(cols)
        return SSThresh, y

    def torch_transform(self, trainer, pl_module, data, label):
        L = pl_module.forward(data)
        S = data - L

        dim = len(L.size()) - 1
        LS = torch.cat((L, S), dim=dim)

        self.num_features = LS.size()[dim]
        self.feature_names = np.array([x for x in range(LS.size(dim))])
        return LS, label
