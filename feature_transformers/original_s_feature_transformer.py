import torch
import numpy as np
import pandas as pd

class OriginalSFeatureTransformer():
    def __init__(self):
        super().__init__()

    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        L = pl_module.forward(torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device))
        S = torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device) - L
        original_S = np.concatenate((data_df.iloc[:,1:-1], S.detach().cpu().numpy()), axis=1)

        y = data_df.iloc[:,-1]

        self.num_features = original_S.shape[1]
        cols = list(data_df.columns[1:-1])
        cols = cols + [f's_{c}' for c in cols]
        self.feature_names = np.array(cols)
        return original_S, y

    def torch_transform(self, trainer, pl_module, data, label, threshold=0.01):
        L = pl_module.forward(data)
        S = data - L

        dim = len(S.size()) - 1
        original_S = torch.cat((data, S), dim=dim)

        self.num_features = original_S.size()[dim]
        self.feature_names = np.array([x for x in range(original_S.size(dim))])
        return original_S, label
