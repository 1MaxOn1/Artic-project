import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F

class Visualize:
    def __init__(self, num_weeks:int = 5):
        self.num_weeks = num_weeks

    def visualize(self, pred, label, start = 0,):
        if pred.device == 'cpu':
            pred = pred.to(torch.device('cpu'))
            label = label.to(torch.device('cpu'))
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        mask = self._get_mask()
        fig, axes = plt.subplots(2, self.num_weeks, figsize = (17, 8))
        fig.text(0.5, 0.93, 'Ground truth', fontsize = 16, ha='center', va='center')
        fig.text(0.5, 0.5, 'Model predictions', fontsize = 16)
        for i in range(start, self.num_weeks + start):
            axes[0, i].imshow(mask - label[i], cmap='Blues')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            for spine in axes[0, i].spines.values():
                spine.set_visible(False)
            axes[0, i].set_title(f'week is month {i//4 + 1}, year is 2020')
            axes[0, i].set_xlabel(f'Mae is {F.l1_loss(torch.tensor(pred[i]), 
                                                      torch.tensor(label[i]))}')
            
            axes[1, i].imshow(mask - pred[i], cmap='Blues')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            for spine in axes[1, i].spines.values():
                spine.set_visible(False)
            axes[1, i].set_title(f'week is month {i//4 + 1}, year is 2020')
            axes[1, i].set_xlabel(f'MAE is {F.l1_loss(torch.tensor(pred[i]),
                                                       torch.tensor(label[i])).item()}')
        plt.subplots_adjust(top=0.9, hspace=0.5)
        plt.show()

    def _get_mask(self, path:str = r'D:\Arctic_project\dataset\land_mask.npy'):
        return np.load(Path(path))