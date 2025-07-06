from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import os

class MakeTimeSeriesPerWeek:
    def __init__(self, path:str):
        self.path = Path(path)
        self.x_train = dict()
        self.y_train = dict()
        self.x_val = dict()
        self.y_val = dict()
        self.x_test = dict()
        self.y_test = dict()
        self.train = list()
        self.val = list()
        self.test = list()
    
    def _get_day(self, filename:str):
        return int(filename.split('_')[1].split('.')[0][-2:])

    def _get_year(self, filename:str):
        return int(filename.split('_')[1].split('.')[0][:4])
    
    def _get_month(self, filename:str):
        return int(filename.split('_')[1].split('.')[0][4:6])
    
    def _normalize(self, ar:np.array):
        return ar / ar.max().item()
    
    def _base_normalize(self, arr:np.array):
        return arr / 100
    
    def _form_data(self, inp_path:list):
        sum_im = None
        for im in inp_path:
            current = np.load(im)
            current = self._base_normalize(current)
            if sum_im is None:
                sum_im = current
            else:
                sum_im += current
        return sum_im / 7
    
    def _collect_average_info(self):
        feat_and_target = list()
        current = list()
        week = list()
        for file in tqdm(sorted(os.listdir(self.path))):
            day = self._get_day(file)
            month = self._get_month(file)
            if day > 28 and month == 2:
                continue
            else:
                week.append(self.path / Path(file))
                if len(week) == 7:
                    val = torch.tensor(self._form_data(week))
                    current.append(val)
                    week = list()
                    if len(current) == 72:
                        feat_and_target.append(current)
                        current = list()
        self.train = feat_and_target[:-2]
        self.val = feat_and_target[-2:-1]
        self.test = [feat_and_target[-1]]

    def _prepare_data(self, inp:list, feat:dict, target:dict):
        for val in tqdm(inp):
            feat[len(feat.keys())] = torch.stack(val[:48])
            target[len(target.keys())] = torch.stack(val[48:])

    def load(self):
        self._collect_average_info()
        self._prepare_data(self.train, self.x_train, self.y_train)
        self._prepare_data(self.val, self.x_val, self.y_val)
        self._prepare_data(self.test, self.x_test, self.y_test)