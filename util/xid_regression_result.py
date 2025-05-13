import pandas as pd
import numpy as np
from sklearn import metrics
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import shutil
from util.argparser import get_args
from itertools import chain





def print_xid_results(xids, dir_path, suffix, graph_type='',rename= True, save= False, args=None, best_result_flag=True, verbose= True):
    df_list =[]
    if best_result_flag:
        filename = 'best_eval_result.csv'
    else:
        filename = 'eval_threshold.csv'

    if graph_type=='':
        graph_suffix = ''
    else:
        graph_suffix = '_'+graph_type
    for xid in xids:
        if rename:
            if args.model == 'gdn':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect(graph'+graph_suffix+')/'+filename)
            elif args.model == 'gat':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect(graph)/'+filename)
            elif args.model == 'fmtm':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect(profile_graph)/'+filename)
            elif args.model == 'fm':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect(profile_graph'+graph_suffix+')/'+filename)
            elif args.model == 'vae':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect/'+filename)
            else:
                raise Exception('{} model not implemented'.format(args.model))
        else:
            if args.model == 'gdn':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect(graph'+graph_suffix+')/'+filename)
            elif args.model == 'gat':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect(graph)/'+filename)
            elif args.model == 'fmtm':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect(profile_graph)/'+filename)
            elif args.model == 'fm':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect(profile_graph'+graph_suffix+')/'+filename)
            elif args.model == 'vae':
                filepath = os.path.join(dir_path, 'xid'+str(xid),suffix,'scores/anomaly_detect/'+filename)
            else:
                raise Exception('{} model not implemented'.format(args.model))

        df = pd.read_csv(filepath, index_col=0)
        if best_result_flag:
            df = df.T

        df_list.append(df)
        if rename:
            os.rename(os.path.join(dir_path, str(xid)), os.path.join(dir_path, f'{xid}_'+suffix))
    merged_df = pd.concat(df_list)
    if best_result_flag:
        merged_df['xid'] = xids
    else:
        num_rows = len(merged_df)
        duplicated_num_per_xid = num_rows//len(xids)
        merged_df['xid'] = list(chain.from_iterable([[i]*duplicated_num_per_xid for i in xids]))
    if verbose:
        print(merged_df.to_markdown())
    if save:
        if not os.path.exists(os.path.join(dir_path, 'sensitive_analysis')):
            os.makedirs(os.path.join(dir_path, 'sensitive_analysis'))
        merged_df.to_csv(os.path.join(dir_path, 'sensitive_analysis',suffix+'.csv'))
    return merged_df

def clear_xid_results(xids, dir_path, suffix, level):
    for xid in xids:
        if level == 'full':
            shutil.rmtree(os.path.join(dir_path, str(xid)+'_'+suffix))

def print_results(dir_path, suffix, graph_type='',rename= True, save= False, args=None):
    df_list =[]
    if graph_type=='':
        graph_suffix = ''
    else:
        graph_suffix = '_'+graph_type

    if args.model == 'gdn':
        filepath = os.path.join(dir_path, suffix,'scores/anomaly_detect(graph'+graph_suffix+')/best_eval_result.csv')
    elif args.model == 'vae':
        filepath = os.path.join(dir_path, suffix,'scores/anomaly_detect/best_eval_result.csv')
    else:
        filepath = os.path.join(dir_path, suffix,'scores/anomaly_detect(profile_graph'+graph_suffix+')/best_eval_result.csv')
    df = pd.read_csv(filepath, index_col=0)
    df = df.T
    df_list.append(df)

    merged_df = pd.concat(df_list)
    print(merged_df.to_markdown())
    if save:
        if not os.path.exists(os.path.join(dir_path, 'sensitive_analysis')):
            os.makedirs(os.path.join(dir_path, 'sensitive_analysis'))
        merged_df.to_csv(os.path.join(dir_path, 'sensitive_analysis',suffix+'.csv'))
    return merged_df

def clear_results( dir_path, suffix, level):
    if level == 'full':
        shutil.rmtree(os.path.join(dir_path, suffix))


def cache_xid_procedure(tol_min = 10, graph_type='',xids =  [13,31,43,45,48,61,62,63,64,74,79]):
    args = get_args()
    dir_path = args.load_model_path
    dir_path = os.path.dirname(dir_path)
    if tol_min != None:
        suffix = str(tol_min)+'mintol'
    else:
        suffix = 'norm'
    print_xid_results(xids, dir_path, suffix, rename= False, save= True, args=args, graph_type=graph_type)

def cache_procedure(tol_min = 10, graph_type=''):
    args = get_args()
    dir_path = args.load_model_path
    dir_path = os.path.dirname(dir_path)
    if tol_min != None:
        suffix = str(tol_min)+'mintol'
    else:
        suffix = 'norm'
    print_results(dir_path, suffix, rename= False, save= True, args=args, graph_type=graph_type)
    clear_results(dir_path, suffix=suffix, level='full')

def print_regression_results(xids, dir_path, suffix, rename= False, save= False, args=None):
    df_list =[]
    
    for xid in xids:
        if rename:
            filepath = os.path.join(dir_path, str(xid),'scores/regression/normalized_metrics_r2.csv')
        else:
            filepath = os.path.join(dir_path, str(xid)+'_'+suffix,'scores/regression/normalized_metrics_r2.csv')
        df = pd.read_csv(filepath, index_col=0)
        mean_values = df[['train','val','test']].mean(axis=0)
        df = pd.DataFrame([mean_values], columns=['train', 'val', 'test'])
        df_list.append(df)
        if rename:
            os.rename(os.path.join(dir_path, str(xid)), os.path.join(dir_path, f'{xid}_'+suffix))
    merged_df = pd.concat(df_list)
    merged_df['xid'] = xids
    print(merged_df.to_markdown())


