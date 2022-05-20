import numpy as np 
import pickle
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd 


class OpponentVAEDataset(Dataset):
    def __init__(self, data_file, player=None):
        self.player = player
        self.data_s,self.data_a,self.data_i = self.read_pickle(data_file)
        
    def __getitem__(self, index):
        return self.data_s[index].astype(np.float32),self.data_a[index],self.data_i[index].astype(np.float32)

    def __len__(self):
        return len(self.data_i)

    def read_pickle(self,data_file):
        data = pd.read_pickle(data_file)

        if self.player is not None:
            data_p = np.array(data['data_p'])
            indices = data_p == self.player
            data_s = np.array(data['data_s'])[indices]
            data_a = np.array(data['data_a'])[indices]
            data_i = np.array(data['data_i'])[indices]
        else:
            data_s = np.array(data['data_s'])
            data_a = np.array(data['data_a'])
            data_i = np.array(data['data_i'])

        return data_s,data_a,data_i


