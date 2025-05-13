import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from util.preprocess import construct_data
import pandas as pd

def slide_rang(data, slide_win, slide_stride, pred_win, mode):
        is_train = True

        flag_reconstruct = slide_win == 1 and slide_stride == 1 and pred_win == 1
        instance_groups = data.groupby('instance',sort=False)
        range_list = []
        for name, group in instance_groups:
            if type(data) == pd.DataFrame:
                total_time_len, node_num = group.shape
            else:
                node_num, total_time_len = group.shape
            if total_time_len == 0:
                continue
            if flag_reconstruct:
                idx_rang = range(total_time_len)
            else:
                if is_train:
                    idx_rang = range(slide_win, total_time_len - pred_win +1, slide_stride)
                else:
                    idx_rang = range(slide_win, total_time_len - pred_win +1)
            
            rang = []
            for idx in idx_rang:
                if flag_reconstruct:
                    rang.append([group.index[idx]])
                else:
                    rang.append([group.index[i] for i in range(idx-slide_win, idx+pred_win)])

            range_list.append(rang)
        if mode == 'test_ignoresync':
            result = []
            for r in range_list:
                result.extend(r)
            return np.array(result)
        else:
            result = []
            for elements in zip(*range_list):
                for list_ in elements:
                    result.append(list_)
            return np.array(result)



def cal_rang_hnode(num1, num2, padding = False):
    """
    Creates a nested NumPy array (or a list of NumPy arrays) with ranges of numbers.

    Args:
        num1: The total number of elements.
        num2: The target length of each sub-array (except potentially the last).

    Returns:
        A 2D NumPy array with the last group padded to be the same length as the previous sub-arrays
    """

    full_chunks = num1 // num2
    remainder = num1 % num2
    
    result = []
    start = 0
    if padding:
        if remainder == 0:
            for i in range(full_chunks):
                end = start + num2
                result.append(np.arange(start, end))
                start = end
            return np.stack(result)
            
        else:
            for i in range(full_chunks):
                end = start + num2
                result.append(np.arange(start, end))
                start = end  
            last_array = np.arange(start, num1)
            padding_length = num2 - len(last_array)
            padded_last = np.pad(last_array, (0, padding_length), 'constant', constant_values=(-1))
            result.append(padded_last)
            return np.stack(result)
    else:
        for i in range(full_chunks):
            end = start + num2
            result.append(np.arange(start, end))
            start = end
        return np.stack(result)

def generate_status_mask(df):
        active_status = ['draining', 'mixed','allocated', 'completing', 'completing*', 'mixed*', 'reserved', 'draining*']
        return df['status'].apply(lambda x: 1 if x in active_status else 0)

class TimeDataset2D(Dataset):
    def __init__(self, df, feature_map, edge_index, mode='train', config = None,timestamp = None):
        self.df = df.copy()
        self.feature_map = feature_map
        self.feature_map_len = len(feature_map)
        self.config = config
        self.edge_index = edge_index.long()
        self.mode = mode
        self.flag_reconstruct = self.config.slide_win == 1 and self.config.slide_stride == 1  and self.config.pred_win==1
        self.timestamp = timestamp
        self.rang = None
        self.process(df)
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: x.timestamp())
        self.status_mask = None
        if 'status' in self.df.columns:
            self.status_mask = torch.from_numpy(generate_status_mask(self.df).values.astype(np.float64))
        self.df_np = torch.from_numpy(self.df[feature_map+['instance_id','label','timestamp']].values)
        
        self.num_hnodes = self.config.hnodes_size
        self.rang_hnode = cal_rang_hnode(self.rang.shape[0],self.num_hnodes)
        del self.df

    def __len__(self):
        return len(self.rang_hnode)


    def process(self, data):
        x_arr, y_arr = [], []
        labels_arr = []

        self.slide_win = self.config.slide_win
        self.slide_stride = self.config.slide_stride
        self.pred_win = self.config.pred_win

        self.rang = slide_rang(data, self.slide_win, self.slide_stride, self.pred_win, self.mode)
        
    def __getitem__(self, idx):
        
        h_win = self.rang_hnode[idx]
        i_win = self.rang[h_win]
        edge_index = self.edge_index
        job_id = 0
        rack_id = 0
        job_group =np.nan
        status_mask =np.nan
        if self.flag_reconstruct:
            feature = self.df_np[i_win[0],:self.feature_map_len]
            if self.status_mask is not None:
                status_mask = self.status_mask[i_win[0]]
            y = feature
            label = self.df_np[i_win[0],-2]
            timestamp = self.df_np[i_win[0],-1]
            instance_id = self.df_np[i_win[0],-3]
        else:
            feature = self.df_np[i_win[:,:-self.pred_win],:self.feature_map_len].permute(0, 2, 1)

            y = self.df_np[i_win[:,-self.pred_win:],:self.feature_map_len]
            label = self.df_np[i_win[:,-self.pred_win:],-2]
            timestamp = self.df_np[i_win[:,-self.pred_win:],-1]
            instance_id = self.df_np[i_win[:,:-self.pred_win],-3]
        return feature,y,label,edge_index,timestamp,instance_id,status_mask,job_group,job_id,rack_id




