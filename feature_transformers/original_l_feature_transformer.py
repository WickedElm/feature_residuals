import torch
import numpy as np
import pandas as pd

class OriginalLFeatureTransformer():
    def __init__(self):
        super().__init__()

    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        L = pl_module.forward(torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device))
        original_L = np.concatenate((data_df.iloc[:,1:-1], L.detach().cpu().numpy()), axis=1)

        y = data_df.iloc[:,-1]

        self.num_features = original_L.shape[1]
        cols = list(data_df.columns[1:-1])
        cols = cols + [f'l_{c}' for c in cols]
        self.feature_names = np.array(cols)
        return original_L, y

    def torch_transform(self, trainer, pl_module, data, label):
        L = pl_module.forward(data)

        dim = len(L.size()) - 1
        original_L = torch.cat((L, data), dim=dim)

        self.num_features = original_L.size()[dim]
        self.feature_names = np.array([x for x in range(original_L.size(dim))])
        return original_L, label
