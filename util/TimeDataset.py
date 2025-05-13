import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from util.preprocess import construct_data
import pandas as pd

def slide_rang(data, slide_win, slide_stride, pred_win, mode):
        is_train = mode == True
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

def generate_status_mask(df):
        active_status = ['draining', 'mixed','allocated', 'completing', 'completing*', 'mixed*', 'reserved', 'draining*']
        return df['status'].apply(lambda x: 1 if x in active_status else 0)

class TimeDataset(Dataset):
    def __init__(self, df, feature_map, edge_index, mode='train', config = None,timestamp = None):
        self.df = df.copy()
        self.feature_map = feature_map
        self.feature_map_len = len(feature_map)
        self.config = config
        self.edge_index = edge_index.long()
        self.mode = mode
        self.flag_reconstruct = self.config.slide_win == 1 and self.config.slide_stride == 1 and self.config.pred_win==1
        self.flag_hnode_graph = hasattr(self.config, 'job_id_col') and config.model == 'hgdn'
        self.timestamp = timestamp
        self.rang = None
        self.process(df)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp']).astype(np.int64) / 10**9
        self.status_mask = None
        if 'status' in self.df.columns:
            status_mask_np = generate_status_mask(self.df).values.astype(np.float64)
            self.status_mask = torch.from_numpy(status_mask_np.copy())
        if self.flag_hnode_graph:
            df_np = self.df[feature_map+['hnode_graph','hnode_id','instance_id','label','timestamp']].values
            self.df_np = torch.from_numpy(df_np.copy())
        else:
            df_np = self.df[feature_map+['instance_id','label','timestamp']].values
            self.df_np = torch.from_numpy(df_np.copy())
        del self.df

    def __len__(self):
        return len(self.rang)

    def process(self, data):
        self.slide_win = self.config.slide_win
        self.slide_stride = self.config.slide_stride
        self.pred_win = self.config.pred_win
        self.rang = slide_rang(data, self.slide_win, self.slide_stride, self.pred_win, self.mode)
        
    def __getitem__(self, idx):
        i_win = self.rang[idx]
        edge_index = self.edge_index
        hnode_id = np.nan
        hnode_group = np.nan
        status_mask = np.nan
        if self.flag_reconstruct:
            feature = self.df_np[i_win[0],:self.feature_map_len]
            y = feature
            label = self.df_np[i_win[0],-2]
            timestamp = self.df_np[i_win[0],-1]
            instance_id = self.df_np[i_win[0],-3]
            if self.flag_hnode_graph:
                hnode_id = self.df_np[i_win[0],-4]
                hnode_group = self.df_np[i_win[0],-5]
        else:
            feature = self.df_np[i_win[:-self.pred_win],:self.feature_map_len].T
            y = self.df_np[i_win[-self.pred_win:],:self.feature_map_len]
            label = self.df_np[i_win[-self.pred_win:],-2]
            timestamp = self.df_np[i_win[-self.pred_win:],-1]
            instance_id = self.df_np[i_win[0],-3]
            if self.flag_hnode_graph:
                hnode_id = self.df_np[i_win[-self.pred_win:],-4]
                hnode_group = self.df_np[i_win[0],-5]

        return feature,y,label,edge_index,timestamp,instance_id,status_mask,hnode_group,hnode_id




