import torch
import numpy as np
import pandas as pd
import feature_transformers.feature_utilities

class LSThresholdFeatureTransformer():
    def __init__(self):
        super().__init__()

    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        L = pl_module.forward(torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device))
        S = torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device) - L
        S = feature_transformers.feature_utilities.threshold_mse(S, threshold=threshold)
        #S = feature_transformers.feature_utilities.threshold_round_ohe(S, threshold=0.9, round_type='up')
        #S = feature_transformers.feature_utilities.threshold_round_ohe(S, threshold=threshold, round_type='down')
        #S = feature_transformers.feature_utilities.threshold_round_continuous(S, threshold=threshold, round_type='down')
        LS = np.concatenate((L.detach().cpu().numpy(), S.detach().cpu().numpy()), axis=1)

        y = data_df.iloc[:,-1]

        self.num_features = LS.shape[1]
        original_cols = list(data_df.columns[1:-1])
        cols = [f'l_{c}' for c in original_cols] + [f's_{c}' for c in original_cols]
        self.feature_names = np.array(cols)
        return LS, y

    def torch_transform(self, trainer, pl_module, data, label, threshold=0.01):
        L = pl_module.forward(data)
        S = data - L
        S = feature_transformers.feature_utilities.threshold_mse(S, threshold=threshold)

        dim = len(L.size()) - 1
        LS = torch.cat((L, S), dim=dim)

        self.num_features = LS.size()[dim]
        self.feature_names = np.array([x for x in range(LS.size(dim))])
        return LS, label
