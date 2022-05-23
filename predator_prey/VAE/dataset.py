import numpy as np 
import pickle
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd 


class OpponentVAEDataset(Dataset):
    def __init__(self, data_file):

        self.data_s,self.data_a,self.data_i = self.read_pickle(data_file)
        
    def __getitem__(self, index):
        # print (self.data_s[index].astype(np.float32),np.argmax(self.data_a[index].astype(np.float32), axis=0),self.data_i[index])
        return self.data_s[index].astype(np.float32),np.argmax(self.data_a[index].astype(np.float32), axis=0),self.data_i[index]

    def __len__(self):
        return len(self.data_i)

    def read_pickle(self,data_file):
        data = pd.read_pickle(data_file)
        # print (data)
        data_s = np.array(data['data_s'])
        data_a = np.array(data['data_a'])
        data_i = np.array(data['data_i'])

        return data_s,data_a,data_i


