import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np


def read_result(saved_dir, saveflag='anomaly_detect'):
    result = pd.read_csv(saved_dir+'/scores/'+saveflag+'/eval_threshold.csv')
    return result

saved_dir_paths = [
    "results/prodigy_mini_vae_10-02--19-46-48/test",
    "results/prodigy_mini_vae_07-17--15-09-19/vae",
    "results/prodigy_mini_gat_07-29--23-15-45/gat",
    "results/prodigy_mini_gdn_07-29--23-13-51/gdn",
    "results/prodigy_mini_fm_07-29--23-08-13/fm",
]
models = [
    "Prodigy(Semi-supervised)",
    "Prodigy(Unsupervised)",
    "GAT",
    "GDN",
    "Mantis",
]
df_all = pd.DataFrame()
df_instance_all = pd.DataFrame()
for saved_dir_path, model in zip(saved_dir_paths, models):
    if model == "GAT":
        df = read_result(saved_dir_path, saveflag='anomaly_detect(graph)')
    elif model == "GDN":
        df = read_result(saved_dir_path, saveflag='anomaly_detect(graph_topk)')
    elif model == "Mantis":
        df = read_result(saved_dir_path, saveflag='anomaly_detect(profile_graph_topk)')
    elif model == "Prodigy(Semi-supervised)":
        df = pd.read_csv(saved_dir_path+'/scores/anomaly_detect/best_eval_result.csv',index_col=0).T
    else:
        df = read_result(saved_dir_path, saveflag='anomaly_detect')
    if model != "Prodigy(Semi-supervised)":
        df = df[df['threshold_per']==0.9999]
    df['model'] = model
    df = df[['model', 'f1', 'tp', 'fp']]
    df_all = pd.concat([df_all, df])
df_all['f1'] = df_all['f1'].apply(lambda x: round(x, 2))
df_all['model'] = pd.Categorical(df_all['model'], categories=models, ordered=True)
df_all = df_all.sort_values(by=['model']).reset_index(drop=True)
print('--- Anomaly Detection Performance on Sandia Dataset ---')
print(df_all.to_markdown(index=False))
