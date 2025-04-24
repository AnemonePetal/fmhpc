import sys
import os
from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
from util.postprocess import align_df
from util.Data import filter_df
from util.TimeDataset import slide_rang
from util.anomaly_dtask import anomaly_score
from util.save import dtask_path
import numpy as np
from util.data_loader import set_dataloader

import pandas as pd
import ast
import json5

def load_old_args(args):
    with open(os.path.dirname(args.paths['test_re'])+'/args.json') as f:
        data_args = json5.load(f)
        for key in data_args:
            setattr(args, key, data_args[key])
    return 0


def retrieve_data(args):
    args.no_store=True
    prepare_env(args)
    args.slide_stride = 1
    args.scaler_only_fit = True
    if args.attack:
        args.retain_beforeattack = True
    data = Dataset(args)

    if  os.path.exists(args.paths['train_re']):
        data.train_re = pd.read_csv(args.paths['train_re'])
        data.train_re['timestamp'] = pd.to_datetime(data.train_re['timestamp'])
        data.train_re = filter_df(data.train_re, args.data_filter)
    else:
        raise Exception('train_re not found')
    
    if  os.path.exists(args.paths['test_re']):
        data.test_re = pd.read_csv(args.paths['test_re'])
        data.test_re['timestamp'] = pd.to_datetime(data.test_re['timestamp'])
        data.test_re = filter_df(data.test_re, args.data_filter)
    else:
        raise Exception('test_re not found')

    if  os.path.exists(args.paths['val_re']):
        data.val_re = pd.read_csv(args.paths['val_re'])
        data.val_re['timestamp'] = pd.to_datetime(data.val_re['timestamp'])
        data.val_re = filter_df(data.val_re, args.data_filter)
    else:
        raise Exception('val_re not found')
    
    set_dataloader(data,args)

    data.train = align_df(data.train_re, data.train, data.train_dataset.rang,args)
    data.test = align_df(data.test_re, data.test, data.test_dataset.rang,args)
    data.val = align_df(data.val_re, data.val, data.val_dataset.rang,args)
    
    if  'status' in data.train.columns:
        data.train = data.train[data.train_re.columns.tolist()+['status']]
        data.val = data.val[data.val_re.columns.tolist()+['status']]
        if args.attack:
            data.test = data.test[data.test_re.columns]
        else:
            data.test = data.test[data.test_re.columns.tolist()+['status']]
    else:
        data.train = data.train[data.train_re.columns]
        data.val = data.val[data.val_re.columns]
        data.test = data.test[data.test_re.columns]

    if args.scaler_only_fit:
        data.train_re= data.scaler.inverse(data.train_re,args.features)
        data.test_re= data.scaler.inverse(data.test_re,args.features)
        data.val_re= data.scaler.inverse(data.val_re,args.features)
    
    data.train = pd.merge(data.train, data.train_re, on=['timestamp', 'instance','label'], how='left', suffixes=('', '_re'))
    data.val = pd.merge(data.val, data.val_re, on=['timestamp', 'instance','label'], how='left', suffixes=('', '_re'))
    data.test = pd.merge(data.test, data.test_re, on=['timestamp', 'instance','label'], how='left', suffixes=('', '_re'))
    

    args.paths['profile_graph_csv'] = os.path.dirname(args.paths['graph_csv']) + '/profile_graph.csv'
    result_dir = 'anomaly_detect({})'.format(args.paths['profile_graph_csv'].split('/')[-1].split('.')[0])
    dtask_paths = dtask_path(args,result_dir,build_dir=False)
    data.topk_test_scores = pd.read_csv(dtask_paths['test_top_scores'])
    data.test['deviation'] = np.abs(data.test[args.features].values - data.test[[i+'_re' for i in args.features]].values).mean(axis=1)
    return data