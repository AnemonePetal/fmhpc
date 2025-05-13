import sys
import os
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from util.xid_regression_result import print_xid_results
from util.argparser import get_args
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset, sort_wrapper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tol_mins = [10]
xids = [63,74,79]
models = ['gdn','gat','fm','vae']
models_path = [
    'results/olcfcutsec_gdn_05-30--01-49-01/best_05-30--01-49-01.pt',
    'results/olcfcutsec_gat_07-08--20-18-18/best_07-08--20-18-18.pt',
    'results/olcfcutsec_fm_05-14--00-59-47/best_05-14--00-59-47.pt',
    'results/olcfcutsec_vae_12-26--19-54-28/model-weights.h5'
]

def extract_result():
    df_list = []
    for tol_min in tol_mins:
        for i, model in enumerate(models):
            args = get_args()
            args.model = model
            args.load_model_path = models_path[i]
            dataset_path='data/olcfcutsec'
            dir_path = args.load_model_path
            dir_path = os.path.dirname(dir_path)
            if tol_min != None:
                suffix = str(tol_min)+'mintol'
            else:
                suffix = 'norm'
            df = print_xid_results(xids, dir_path, suffix, rename= False, save= False, args=args, graph_type='topk', best_result_flag= False, verbose=False)
            df['tol_min'] = tol_min
            df['model'] = args.model
            df_list.append(df)
    df = pd.concat(df_list)
    df = df.reset_index()
    return df


df = extract_result()
fm_rows = pd.DataFrame(df[df['model']=='fm'],columns=df.columns)
fm_rows['model'] = 'fm'
fm_rows['f1'] = fm_rows['f1_graph']
fm_rows['tp'] = fm_rows['tp_graph']
fm_rows['fp'] = fm_rows['fp_graph']
df = pd.concat([df[df['model']!='fm'], fm_rows])
df.reset_index(drop=True, inplace=True)
df = df[['xid', 'tol_min','threshold_per', 'model', 'f1', 'tp', 'fp']]


df2 = df[df['threshold_per'].isin([0.99,0.999,0.9999])]
df2 = df2.groupby(['xid','tol_min','model']).apply(lambda x: x[x['f1']==x['f1'].max()].drop_duplicates(subset=['f1'])).reset_index(drop=True)
df2['model'] = df2['model'].replace('vae', 'Prodigy')
df2['model'] = df2['model'].replace('fm', 'Mantis')
df2['model'] = df2['model'].replace('gdn', 'GDN')
df2['model'] = df2['model'].replace('gat', 'GAT')

models =['Prodigy','GAT','GDN','Mantis']
df2['model'] = pd.Categorical(df2['model'], categories=models, ordered=True)
df2 = df2.sort_values(by=['xid', 'tol_min', 'model']).reset_index(drop=True)


titles = ["ECC page retirement or row remapping","NVLINK Error","GPU Off Bus"]
print('--- Anomaly Detection Performance on OLCF Real-world Anomalies Dataset ---')
for i, xid in enumerate(xids):
    print(f'--- {titles[i]} ---')
    df_xid = df2[df2['xid']==xid][['model','f1','tp','fp']]
    
    print(df_xid.to_markdown(index=False))
    