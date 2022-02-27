import torch
import numpy as np
import pandas as pd

class OriginalFeatureTransformer():
    def __init__(self):
        super().__init__()

    # Returns data along with associated labels (if any)
    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        self.num_features = data_df.shape[1] - 2
        self.feature_names = data_df.columns[1:-1]
        return data_df.iloc[:,1:-1], data_df.iloc[:,-1]

    def torch_transform(self, trainer, pl_module, data, label, threshold=0.01):
        dim = len(data.size()) - 1

        self.num_features = data.size()[dim]
        self.feature_names = np.array([x for x in range(data.size(dim))])
        return data, label
