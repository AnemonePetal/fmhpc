import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from torchinfo import summary

from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
import numpy as np

from util.Data import Dataset
from models.GDN import GDN_wrapper
from models.GAT import GAT_wrapper
from models.FM import FM_wrapper
from models.FM_NM import FM_NM_wrapper
from copy import copy
import pandas as pd

def rename_model_name(model_name):
    if model_name == 'fm':
        model_name = 'SysMixer'
    elif model_name == 'gat':
        model_name = 'GAT'
    elif model_name == 'gdn':
        model_name = 'GDN'
    elif model_name == 'fmnm':
        model_name = 'SysMixer-NM'
    return model_name


def summary_model(model, data, model_name, print_total_params_only=False):
    if model_name == 'fmnm':
        x, y, labels, edge_index, timestamp, instance_id, status_mask, _ ,_,_ = next(iter(data.test_dataloader))
    else:
        x, y, labels, edge_index, timestamp, instance_id, status_mask, _ ,_ = next(iter(data.test_dataloader))
    
    device = next(model.parameters()).device
    x, y = [item.to(device).float() for item in [x, y]]
    if edge_index is not None:
        edge_index = edge_index.to(device).float()
    if instance_id is not None:
        instance_id = instance_id.to(device).int()

    input_data = []
    if model_name in ['fm']:
        input_data = [x]
    elif model_name in ['gdn','gat']:
        input_data = [x, edge_index]
    elif model_name in ['fmnm']:
         input_data = [x, instance_id]
    if not input_data:
        print(f"Warning: Input data not determined for model type {model_name}. Cannot generate summary.")
        return
    model_summary = summary(model, input_data=input_data, verbose=0)

    if print_total_params_only:
        print(f"Total Parameters: {model_summary.total_params}")
    else:
        print(model_summary)

    return model_summary.total_params

def model_selector(args,data):
    if args.model == 'gdn':
        model = GDN_wrapper(args,data,tensorboard=False)
    elif args.model == 'gat':
        model = GAT_wrapper(args,data,tensorboard=False)
    elif args.model == 'fm':
        model = FM_wrapper(args,data,tensorboard=False)
    elif args.model == 'fmnm':
        args.hnodes_size = data.train.instance.nunique()
        model = FM_NM_wrapper(args,data,tensorboard=False)
    return model

def get_rmse(metric='rmse' , savepath="",inverse_scaler=None,verbose=True):
    inverse_scaler_str = 'normalized' if inverse_scaler is None else 'original'
    if os.path.exists(savepath+ inverse_scaler_str + "_metrics_"+metric+".csv"):
        feature_metric = pd.read_csv(savepath + inverse_scaler_str + "_metrics_"+metric+".csv")
        if verbose:
            data_type ='test'
            stat_metric = np.mean(feature_metric[data_type])
            print(f'RMSE ({data_type}) : {stat_metric}')
        return feature_metric
    else:
        print('File not found: '+savepath+ inverse_scaler_str + "_metrics_"+metric+".csv")
        print('Returning dataframe with zeros')
        return pd.DataFrame([[0, 0, 0]], columns=['train','val','test'])

if __name__=='__main__':
    dataset_list = ['jlab','olcfcutsec']
    model_list = [['gat','gdn','fmnm','fm']]* len(dataset_list)
    load_model_paths = [
        [
        "results/jlab_gat_03-13--18-58-08/best_03-13--18-58-08.pt",
        "results/jlab_gdn_03-13--16-01-00/best_03-13--16-01-00.pt",
        "results/jlab_fmnm_04-14--04-32-10/best_04-14--04-32-10.pt",
        "results/jlab_fm_03-13--13-35-29/best_03-13--13-35-29.pt",
        ],
        [
        "results/olcfcutsec_gat_07-08--20-18-18/best_07-08--20-18-18.pt",
        "results/olcfcutsec_gdn_05-30--01-49-01/best_05-30--01-49-01.pt",
        "results/olcfcutsec_fmnm_04-14--04-55-00/best_04-14--04-55-00.pt",
        "results/olcfcutsec_fm_05-14--00-59-47/best_05-14--00-59-47.pt",
        ]
    ]
    args_base = get_args()
    args_base.no_store=True
    args_base.reload_args = False
    args_base.dataloader_num_workers = 0
    model_size_list = []
    rmse_list = []
    train_time_list = []
    for i, dataset in enumerate(dataset_list):
        for model_name,load_model_path in zip(model_list[i],load_model_paths[i]):
            args = copy(args_base)
            args.dataset = dataset 
            args.model = model_name
            args.save_subdir = model_name
            args.load_model_path = load_model_path
            prepare_env(args)
            data = Dataset(args)
            model = model_selector(args,data)
            print(f"-----------------Process: {rename_model_name(model_name)} ({dataset})-----------------")
            model_size = summary_model(model.model, data, model_name, print_total_params_only=True)
            rmse = get_rmse(metric='rmse', savepath=args.paths['score_dir']+'regression/', inverse_scaler=None)
            model_size_list.append(model_size)
            rmse_list.append(rmse['test'].mean())

    datasets_for_df = []
    models_for_df = []
    for i, dataset in enumerate(dataset_list):
        for model_name in model_list[i]:
            datasets_for_df.append(dataset)
            models_for_df.append(rename_model_name(model_name))

    df = pd.DataFrame({
        'Dataset': datasets_for_df,
        'Model': models_for_df,
        'Model Size': model_size_list,
        'RMSE': rmse_list
    })
    print("\n--- Summary ---")
    print(df)
