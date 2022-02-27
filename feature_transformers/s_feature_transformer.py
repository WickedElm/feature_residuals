import torch
import numpy as np
import pandas as pd

class SFeatureTransformer():
    def __init__(self):
        super().__init__()

    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        L = pl_module.forward(torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device))
        S = torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device) - L

        y = data_df.iloc[:,-1]

        self.num_features = data_df.shape[1] - 2
        self.feature_names = data_df.columns[1:-1]
        return S.detach().numpy(), y

    def torch_transform(self, trainer, pl_module, data, label, threshold=0.01):
        L = pl_module.forward(data)
        S = data - L

        dim = len(S.size()) - 1

        self.num_features = S.size()[dim]
        self.feature_names = np.array([x for x in range(S.size(dim))])
        return S, label
