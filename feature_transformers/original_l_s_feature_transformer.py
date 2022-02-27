import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import wandb
import math

class OriginalLSFeatureTransformer():
    def __init__(self):
        super().__init__()

    def transform(self, trainer, pl_module, data_df, threshold=0.01):
        L = pl_module.forward(torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device))
        S = torch.tensor(data_df.iloc[:,1:-1].values).float().to(pl_module.device) - L
        original_LS = np.concatenate((data_df.iloc[:,1:-1], L.detach().cpu().numpy(), S.detach().cpu().numpy()), axis=1)

        self.L = L.detach().cpu().numpy()
        self.S = S.detach().cpu().numpy()
        self.original = data_df

        y = data_df.iloc[:,-1]

        self.num_features = original_LS.shape[1]
        cols = list(data_df.columns[1:-1])
        cols = cols + [f'l_{c}' for c in cols] + [f's_{c}' for c in cols]
        self.feature_names = np.array(cols)
        return original_LS, y

    def torch_transform(self, trainer, pl_module, data, label, threshold=0.01):
        L = pl_module.forward(data)
        S = data - L

        dim = len(L.size()) - 1
        original_LS = torch.cat((data, L, S), dim=dim)

        self.num_features = original_LS.size()[dim]
        self.feature_names = np.array([x for x in range(original_LS.size(dim))])
        return original_LS, label

    def plot_lso_comparison(self, results_dir, prefix):
        self.L = np.hstack((self.L, self.original.label.values[:,None]))
        l_benign = self.L[self.L[:,35] == 0]
        l_attack = self.L[self.L[:,35] == 1]

        self.S = np.hstack((self.S, self.original.label.values[:,None]))
        s_benign = self.S[self.S[:,35] == 0]
        s_attack = self.S[self.S[:,35] == 1]

        # Save the actual data
        l_benign_df = pd.DataFrame(l_benign[0:100,:-1], columns=self.original.columns[1:-1])
        l_benign_df.to_csv(f'{results_dir}/{prefix}_l_benign.csv', index=False)
        l_attack_df = pd.DataFrame(l_attack[0:100,:-1], columns=self.original.columns[1:-1])
        l_attack_df.to_csv(f'{results_dir}/{prefix}_l_attack.csv', index=False)
        s_benign_df = pd.DataFrame(s_benign[0:100,:-1], columns=self.original.columns[1:-1])
        s_benign_df.to_csv(f'{results_dir}/{prefix}_s_benign.csv', index=False)
        s_attack_df = pd.DataFrame(s_attack[0:100,:-1], columns=self.original.columns[1:-1])
        s_attack_df.to_csv(f'{results_dir}/{prefix}_s_attack.csv', index=False)
        self.original.loc[self.original.label == 0].iloc[0:100,1:-1].to_csv(f'{results_dir}/{prefix}_original_benign.csv', index=False)
        self.original.loc[self.original.label == 1].iloc[0:100,1:-1].to_csv(f'{results_dir}/{prefix}_original_attack.csv', index=False)

        # Based on the fact that we normalize our data to be between 0 and 1
        vmax = 2
        s_benign_vmax = s_benign[0:100,:-1].max().max()
        s_attack_vmax = s_attack[0:100,:-1].max().max()
        s_vmax = np.max([s_benign_vmax, s_attack_vmax])

        fig, axarr = plt.subplots(3, 2, sharex=True, sharey=True)
        plt.sca(axarr[0,0])
        
        plt.imshow(self.original.loc[self.original.label == 0].iloc[0:100,1:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        plt.ylabel('Original')
        plt.xlabel('Benign Network Flows')
        plt.tight_layout()
        
        plt.sca(axarr[0,1])
        plt.imshow(self.original.loc[self.original.label == 1].iloc[0:100,1:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)
        plt.colorbar()
        ax.xaxis.set_label_position('top')
        plt.xlabel('Attack Network Flows')
        plt.tight_layout()
        
        plt.sca(axarr[1,0])
        plt.imshow(l_benign[0:100,:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('L')
        plt.tight_layout()
        
        plt.sca(axarr[1,1])
        plt.imshow(l_attack[0:100,:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.colorbar()
        plt.tight_layout()
        
        plt.sca(axarr[2,0])
        plt.imshow(s_benign[0:100,:-1], vmin=-s_vmax, vmax=s_vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('S')
        plt.tight_layout()
        
        plt.sca(axarr[2,1])
        plt.imshow(s_attack[0:100,:-1], vmin=-s_vmax, vmax=s_vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.colorbar()
        plt.tight_layout()
        
        # Save plot and close figure
        plt.savefig(f'{results_dir}/{prefix}_lso_comparison.png')
        wandb.log({f'{prefix}/lso_comparison':plt})
        plt.close()

        # Plot a mixture of both attack and benign together
        all_l_df = pd.DataFrame(self.L[0:500], columns=self.original.columns[1:])
        all_s_df = pd.DataFrame(self.S[0:500], columns=self.original.columns[1:])

        # Save data
        self.original.iloc[0:500,1:].to_csv(f'{results_dir}/{prefix}_all_original_benign.csv', index=False)
        all_l_df.to_csv(f'{results_dir}/{prefix}_l_all.csv', index=False)
        all_s_df.to_csv(f'{results_dir}/{prefix}_s_all.csv', index=False)

        fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True)
        plt.sca(axarr[0])
        
        plt.imshow(self.original.iloc[0:500,1:], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        plt.ylabel('Original')
        plt.xlabel('All Network Flows')
        plt.colorbar()
        plt.tight_layout()
        
        plt.sca(axarr[1])
        plt.imshow(all_l_df, vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('L')
        plt.colorbar()
        plt.tight_layout()
        
        plt.sca(axarr[2])
        plt.imshow(all_s_df, vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('S')
        plt.colorbar()
        plt.tight_layout()
        
        # Save plot and close figure
        plt.savefig(f'{results_dir}/{prefix}_all_lso_comparison.png')
        wandb.log({f'{prefix}/all_lso_comparison':plt})
        plt.close()
