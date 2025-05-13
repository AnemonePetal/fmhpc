import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

def update_savepath_datestr(args):
    if not hasattr(args, 'load_model_path') or args.load_model_path == '':
        now = datetime.now()
        args.datestr = now.strftime('%m-%d--%H-%M-%S')
        args.datestr_pt = now.strftime('%m-%d--%H-%M-%S')
    else:
        args.datestr = os.path.dirname(args.load_model_path).split('_')[-1].split('.')[0]
        args.datestr_pt = os.path.basename(args.load_model_path).split('_')[-1].split('.')[0]
        args.model = os.path.dirname(args.load_model_path).split('_')[-2]
        if not os.path.exists(os.path.dirname(args.load_model_path)+ f'/{args.save_subdir}'):
            print(f'[WARNING]: {args.save_subdir} does not exist in {os.path.dirname(args.load_model_path)}')
            print('[WARNING]: it may be created to save results')



def get_save_path(args,build_dir=False):
    
    dir_path = args.dataset.split('/')[0]+'_'+args.model
    datestr = args.datestr
    datestr_pt = args.datestr_pt
    full_dir_path=f'{dir_path}_{datestr}'
    if args.save_subdir != '':
        full_dir_path = f'{full_dir_path}/{args.save_subdir}'
    
    paths = {
        'dataset': f'./data/{args.dataset}',
        'cache_pt': f'./cache/pretrained/{dir_path}/best_{datestr_pt}.pt',
        'best_pt': f'./results/{dir_path}_{datestr}/best_{datestr_pt}.pt',
        'train_re': f'./results/{dir_path}_{datestr}/train_re.csv',
        'val_re': f'./results/{dir_path}_{datestr}/val_re.csv',
        'test_re': f'./results/{full_dir_path}/test_re.csv',
        'loss_csv': f'./results/{full_dir_path}/other/loss.csv',
        'graph_csv': f'./results/{dir_path}_{datestr}/graph.csv',
        'readable_graph_csv': f'./results/{dir_path}_{datestr}/readable_graph.csv',
        'score_dir': f'./results/{full_dir_path}/scores/',
    }


    if build_dir:
        Path(f'./cache/pretrained/{dir_path}').mkdir(parents=True, exist_ok=True)
        Path(f'./results/{dir_path}_{datestr}').mkdir(parents=True, exist_ok=True)
        Path(f'./results/{full_dir_path}').mkdir(parents=True, exist_ok=True)
        Path(f'./results/{full_dir_path}/other').mkdir(parents=True, exist_ok=True)
        Path(f'./results/{full_dir_path}/scores').mkdir(parents=True, exist_ok=True)

    args.paths = paths

def dtask_path(args, task, build_dir=False):
    if not isinstance(task, str):
        raise Exception('task should be a string')
    if task.startswith('anomaly_detect'):
        file_names = ['train_scores', 'val_scores', 'test_scores', 'train_top_scores', 'val_top_scores', 'test_top_scores','best_eval_result','best_eval_instance_result']
    else:
        file_names = []
    paths = {}
    for file_name in file_names:
        paths[file_name] = args.paths['score_dir'] + f'{task}/{file_name}.csv'
    if build_dir:
        Path(args.paths['score_dir'] + f'{task}').mkdir(parents=True, exist_ok=True)
    if paths == {}:
        return args.paths['score_dir'] + f'{task}/'
    else:
        return paths


def save_result(args, data, result_type, save2disk=True):
    if result_type != 'test' and result_type != 'train' and result_type != 'val':
        print('result_type should be test or train or val')
        return

    if result_type == 'train':
        rang = data.train_dataset.rang[:,-args.pred_win:].flatten()
        if 'anomaly_source' in data.train.columns:
            df = data.train[['timestamp','instance','label','anomaly_source']].iloc[rang].copy()
        else:
            df = data.train[['timestamp','instance','label']].iloc[rang].copy()
        if hasattr(data, 'train_re'):
            save_path_l = args.paths['train_re']
            df_l = format_df(data.train_re,df,args.features)
    elif result_type == 'val':
        rang = data.val_dataset.rang[:,-args.pred_win:].flatten()
        if 'anomaly_source' in data.val.columns:
            df = data.val[['timestamp','instance','label','anomaly_source']].iloc[rang].copy()
        else:
            df = data.val[['timestamp','instance','label']].iloc[rang].copy()
        if hasattr(data, 'val_re'):
            save_path_l = args.paths['val_re']
            df_l = format_df(data.val_re,df,args.features)
    elif result_type == 'test':
        rang = data.test_dataset.rang[:,-args.pred_win:].flatten()
        if 'anomaly_source' in data.test.columns:
            df = data.test[['timestamp','instance','label','anomaly_source']].iloc[rang].copy()
        else:
            df = data.test[['timestamp','instance','label']].iloc[rang].copy()
        if hasattr(data, 'test_re'):
            save_path_l = args.paths['test_re']
            df_l = format_df(data.test_re,df,args.features)

    if save2disk:
        df_l.to_csv(save_path_l, index=False)
    return df_l


def save_result_2d(args, data, result_type, save2disk=True):
    if result_type != 'test' and result_type != 'train' and result_type != 'val':
        print('result_type should be test or train or val')
        return

    if result_type == 'train':
        rang = data.train_dataset.rang
        rang_hnode = data.train_dataset.rang_hnode.flatten()
        rang = rang[rang_hnode]
        rang = rang[:,-args.pred_win:].flatten()
        if 'anomaly_source' in data.train.columns:
            df = data.train[['timestamp','instance','label','anomaly_source']].iloc[rang].copy()
        else:
            df = data.train[['timestamp','instance','label']].iloc[rang].copy()
        if hasattr(data, 'train_re'):
            save_path_l = args.paths['train_re']
            if data.train_re.shape[0]!=df.shape[0]:
                raise ValueError('data.train_re.shape[0]!=df.shape[0]')
            df_l = format_df(data.train_re,df,args.features)
    elif result_type == 'val':
        rang = data.val_dataset.rang
        rang_hnode = data.val_dataset.rang_hnode.flatten()
        rang = rang[rang_hnode]
        rang = rang[:,-args.pred_win:].flatten()

        if 'anomaly_source' in data.val.columns:
            df = data.val[['timestamp','instance','label','anomaly_source']].iloc[rang].copy()
        else:
            df = data.val[['timestamp','instance','label']].iloc[rang].copy()
        if hasattr(data, 'val_re'):
            save_path_l = args.paths['val_re']
            if data.val_re.shape[0]!=df.shape[0]:
                raise ValueError('data.val_re.shape[0]!=df.shape[0]')
            df_l = format_df(data.val_re,df,args.features)
    elif result_type == 'test':
        rang = data.test_dataset.rang
        rang_hnode = data.test_dataset.rang_hnode.flatten()
        rang = rang[rang_hnode]
        rang = rang[:,-args.pred_win:].flatten()

        if 'anomaly_source' in data.test.columns:
            df = data.test[['timestamp','instance','label','anomaly_source']].iloc[rang].copy()
        else:
            df = data.test[['timestamp','instance','label']].iloc[rang].copy()
        if hasattr(data, 'test_re'):
            save_path_l = args.paths['test_re']
            if data.test_re.shape[0]!=df.shape[0]:
                raise ValueError('data.test_re.shape[0]!=df.shape[0]')
            df_l = format_df(data.test_re,df,args.features)

    if save2disk:
        df_l.to_csv(save_path_l, index=False)
    return df_l

def format_df(df_re,df,features):
        np_result = np.array(df_re)
        df =df.copy()
        new_df = pd.DataFrame(np_result, columns=features)
        df = pd.concat([df.reset_index(drop=True), new_df], axis=1)
        del np_result
        df = df.reset_index(drop=True)
        return df

def save_graph(graph,features, graph_path ,readable_graph_path):
    if type(graph) != pd.DataFrame:
        df = pd.DataFrame(graph,columns=['source','destination','value'])
        df['source'] = df['source'].apply(lambda x: int(x))
        df['destination'] = df['destination'].apply(lambda x: int(x))
    else:
        df = graph

    df2 = df.copy()
    df2['source'] = df['source'].apply(lambda x: features[x])
    df2['destination'] = df['destination'].apply(lambda x: features[x])
    df.to_csv(graph_path, index=False)
    df2.to_csv(readable_graph_path, index=False)

    df3 = df2.copy()
    df3 = df3[df3['source'] != df3['destination']]
    df3.to_csv(readable_graph_path.replace('.csv','_no_diag.csv'), index=False)