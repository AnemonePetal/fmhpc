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

tol_mins = [1,2,3,4,5,6,7,8,9,10]
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

def matplotlib_lines_best_thr(df, x_col, y_col, color_col, xid, xlabel, ylabel, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    categories = ['Mantis','GDN','GAT','Prodigy']
    colors = {
        'Prodigy': '#655574',
        'GAT': '#5b90b4',
        'GDN': '#e79a56',
        'Mantis': '#cd6465',
    }
    markers = {
        'Prodigy': 'D',
        'GAT': '^',
        'GDN': 'o',
        'Mantis': 'X',
    }
    for category in categories:
        df_category = df[df[color_col] == category]
        ax.plot(df_category[x_col], df_category[y_col], marker=markers[category], label=f'{category}', color=colors[category], linewidth=2, markersize=8)

    ax.set_xlabel(xlabel, fontsize=24)
    ax.grid(True, which='both', linestyle='solid', linewidth=0.5)
    if xid == 63:
        ax.set_ylabel(ylabel, fontsize=24)
        ax.legend(bbox_to_anchor=(-0.01, 1.03), loc='upper left', prop={'size': 22}, ncol=2)
        y_min = 0.47
        y_max = 0.7
    elif xid == 74:
        ax.set_ylabel(ylabel, fontsize=24)
        y_min = 0.45
        y_max = 0.7
    elif xid == 79:
        ax.set_ylabel(ylabel, fontsize=24)
        y_min = 0.42
        y_max = 0.71
    ax.set_yticks(np.linspace(0.5, 0.7, 3))
    ax.yaxis.set_tick_params(labelsize=22)
    ax.xaxis.set_tick_params(labelsize=22)

    ax.set_title(f'{title}', fontsize=24)
    plt.tight_layout(pad=0.1)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

df = extract_result()
fm_rows = pd.DataFrame(df[df['model']=='fm'],columns=df.columns)
fm_rows['model'] = 'fm'
fm_rows['f1'] = fm_rows['f1_graph']
fm_rows['tp'] = fm_rows['tp_graph']
fm_rows['fp'] = fm_rows['fp_graph']
df = pd.concat([df[df['model']!='fm'], fm_rows])

df2 = df[df['threshold_per'].isin([0.99,0.999,0.9999])]
df_best = df2.groupby(['xid','tol_min','model']).apply(lambda x: x[x['f1']==x['f1'].max()].drop_duplicates(subset=['f1'])).reset_index(drop=True)
df_best['model'] = df_best['model'].replace('vae', 'Prodigy')
df_best['model'] = df_best['model'].replace('fm', 'Mantis')
df_best['model'] = df_best['model'].replace('gdn', 'GDN')
df_best['model'] = df_best['model'].replace('gat', 'GAT')
savepaths = ['fig9_a.png','fig9_b.png','fig9_c.png']
titles = ["ECC page retirement or row remapping","NVLINK Error","GPU Off Bus"]
for i, xid in enumerate(xids):
    matplotlib_lines_best_thr(df_best[(df_best['xid']==xid)], 'tol_min', 'f1', 'model', xid, 'Anomaly time window', 'F1', savepaths[i], titles[i])

