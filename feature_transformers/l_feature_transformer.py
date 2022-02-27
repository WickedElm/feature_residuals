import torch
import numpy as np
import pandas as pd

class LFeatureTransformer():
    def __init__(self):
        super().__init__()

    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        L = pl_module.forward(torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device))
        y = data_df.iloc[:,-1]
        self.num_features = data_df.shape[1] - 2
        self.feature_names = data_df.columns[1:-1]
        return L.detach().numpy(), y

    def torch_transform(self, trainer, pl_module, data, label):
        L = pl_module.forward(data)

        dim = len(L.size()) - 1

        self.num_features = L.size()[dim]
        self.feature_names = np.array([x for x in range(L.size(dim))])
        return L, label
