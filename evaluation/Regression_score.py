import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
from util.argparser import get_args
from util.env import prepare_env
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import textwrap

def wrap_labels(ax, width, break_long_words=False, flag_break=':'):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        text_wrap= textwrap.fill(text, width=width,
                      break_long_words=break_long_words)
        labels.append(text_wrap)

    ax.set_xticklabels(labels, rotation=0)

def add_blank_newline_labels(ax,num_newline=1):
    labels = []
    for label in ax.get_xticklabels():
        text_wrap = label.get_text()
        for i in range(num_newline):
            text_wrap += '\n'
        labels.append(text_wrap)
    ax.set_xticklabels(labels, rotation=0)

def rename_model_name(model_name):
    if model_name == 'fm':
        model_name = 'FM'
    elif model_name == 'gat':
        model_name = 'GAT'
    elif model_name == 'gdn':
        model_name = 'GDN'
    elif model_name == 'fmtm':
        model_name = 'Time-Mixing-only'
    elif model_name == 'fmnm':
        model_name = 'FM-NM'
    return model_name

def retrieve_old_feature_score(metric='mse',verbose=False,inverse_scaler=None,savepath='./'):
    inverse_scaler_str = 'normalized' if inverse_scaler is None else 'original'
    if os.path.exists(savepath+ inverse_scaler_str + "_metrics_"+metric+".csv"):
        feature_metric = pd.read_csv(savepath + inverse_scaler_str + "_metrics_"+metric+".csv")
        if verbose:
            print('======** {0} {1} **======'.format(inverse_scaler_str, metric))
            for data_type in ['train','val','test']:
                stat_metric = np.mean(feature_metric[data_type])
                print(f'{data_type} : {stat_metric}')
        return feature_metric
    else:
        print('File not found: '+savepath+ inverse_scaler_str + "_metrics_"+metric+".csv")
        print('Returning dataframe with zeros')
        return pd.DataFrame([[0, 0, 0]], columns=['train','val','test'])

def process_data(rename_model_name, retrieve_old_feature_score, args, dataset_list, model_list, load_model_paths):
    jlab_score_df = pd.DataFrame()
    olcf_socre_df = pd.DataFrame()

    score_stat = pd.DataFrame()
    for dataset, model_name, load_model_paths in zip(dataset_list, model_list,load_model_paths):
        args.no_store=True
        args.dataset = dataset
        args.model = model_name
        args.save_subdir = model_name
        args.load_model_path = load_model_paths
        prepare_env(args)

        rmse_score =retrieve_old_feature_score(metric='rmse' , savepath=args.paths['score_dir']+'regression/')
        test_mean_rmse = np.mean(rmse_score['test'])
        if dataset == 'jlab':
            dataset_label = 'JLab'
        elif dataset == 'olcfcutsec':
            dataset_label = 'OLCF'
        else:
            dataset_label = dataset

        model_name_label = rename_model_name(model_name)
        row = [dataset_label, model_name_label, test_mean_rmse]
        row = pd.Series(row, index=['dataset','model', 'rmse'])
        score_stat = pd.concat([score_stat, row.to_frame().T], ignore_index=True)

        rmse_score.rename(columns={'Unnamed: 0':'features'}, inplace=True)
        if rmse_score.shape[0] == 1:
            continue
        if dataset_label == 'JLab':
            if jlab_score_df.empty:
                jlab_score_df['features'] = rmse_score['features']
            jlab_score_df = pd.concat([jlab_score_df, pd.DataFrame(rmse_score['test'].values,columns=[model_name_label])] ,axis=1)
        elif dataset_label == 'OLCF':
            if olcf_socre_df.empty:
                olcf_socre_df['features'] = rmse_score['features']
            olcf_socre_df = pd.concat([olcf_socre_df, pd.DataFrame(rmse_score['test'].values,columns=[model_name_label])] ,axis=1)


    if not jlab_score_df.empty:
        jlab_score_df['features'] = jlab_score_df['features'].str.replace(r'node_memory_', 'mem: ')
        jlab_score_df['features'] = jlab_score_df['features'].str.replace(r'node_disk_', 'disk: ')
        jlab_score_df['features'] = jlab_score_df['features'].str.replace(r'^(?!mem:|disk:)', 'cpu: ', regex=True)
        jlab_score_df['features'] = jlab_score_df['features'].str.replace(r'_', ' ')

        if 'FM' in jlab_score_df.columns:
            ref_model_rmse = jlab_score_df.set_index('features')['FM']
        else:
             ref_model_rmse = jlab_score_df.set_index('features').mean(axis=1)

        ref_model_rmse = ref_model_rmse.sort_values(ascending=False)
        top_5_idx = ref_model_rmse[:5].index
        bottom_5_idx = ref_model_rmse[-5:].index

        jlab_score_df_indexed = jlab_score_df.set_index('features')

        jlab_score_df_0 = jlab_score_df_indexed.loc[top_5_idx]
        jlab_score_df_stack_0 = jlab_score_df_0.stack().reset_index()
        jlab_score_df_stack_0.rename(columns={'level_1':'model', 0:'RMSE'}, inplace=True)

        jlab_score_df_1 = jlab_score_df_indexed.loc[bottom_5_idx]
        jlab_score_df_stack_1 = jlab_score_df_1.stack().reset_index()
        jlab_score_df_stack_1.rename(columns={'level_1':'model', 0:'RMSE'}, inplace=True)
    else:
        jlab_score_df_stack_0 = pd.DataFrame(columns=['features', 'model', 'RMSE'])
        jlab_score_df_stack_1 = pd.DataFrame(columns=['features', 'model', 'RMSE'])


    if not olcf_socre_df.empty:
        olcf_socre_df['features'] = olcf_socre_df['features'].str.replace(r'_(?!.*_)(\w+)', r'_\1', regex=True)

        if 'FM' in olcf_socre_df.columns:
             ref_model_rmse = olcf_socre_df.set_index('features')['FM']
        else:
             ref_model_rmse = olcf_socre_df.set_index('features').mean(axis=1)

        ref_model_rmse = ref_model_rmse.sort_values(ascending=False)
        top_5_idx = ref_model_rmse[:5].index
        bottom_5_idx = ref_model_rmse[-5:].index

        olcf_socre_df_indexed = olcf_socre_df.set_index('features')

        olcf_socre_df_0 = olcf_socre_df_indexed.loc[top_5_idx]
        olcf_socre_df_stack_0 = olcf_socre_df_0.stack().reset_index()
        olcf_socre_df_stack_0.rename(columns={'level_1':'model', 0:'RMSE'}, inplace=True)

        olcf_socre_df_1 = olcf_socre_df_indexed.loc[bottom_5_idx]
        olcf_socre_df_stack_1 = olcf_socre_df_1.stack().reset_index()
        olcf_socre_df_stack_1.rename(columns={'level_1':'model', 0:'RMSE'}, inplace=True)
    else:
        olcf_socre_df_stack_0 = pd.DataFrame(columns=['features', 'model', 'RMSE'])
        olcf_socre_df_stack_1 = pd.DataFrame(columns=['features', 'model', 'RMSE'])


    model_order = score_stat['model'].unique().tolist()

    return jlab_score_df_stack_0, jlab_score_df_stack_1, olcf_socre_df_stack_0, olcf_socre_df_stack_1, model_order


def format_e(n, mantissa_digits=1):
    """Formats a number in scientific notation with specified mantissa digits
       and without leading zero in the exponent."""
    if pd.isna(n):
        return ""
    try:
        format_string = f"%.{mantissa_digits}e"
        a = format_string % n
        parts = a.split('e')
        mantissa = parts[0]
        exponent_sign = parts[1][0]
        exponent_val = int(parts[1][1:])
        return f"{mantissa}e{exponent_sign}{exponent_val}"
    except (ValueError, IndexError):
        return str(n)

def print_results(jlab_high_df, jlab_low_df, olcf_high_df, olcf_low_df, model_order,savepath='./'):
    jlab_high_df['Dataset'] = 'JLab'
    jlab_high_df['Telemetries'] = 'Telemetries w/ Highest RMSE'
    jlab_low_df['Dataset'] = 'JLab'
    jlab_low_df['Telemetries'] = 'Telemetries w/ Lowest RMSE'
    olcf_high_df['Dataset'] = 'OLCF'
    olcf_high_df['Telemetries'] = 'Telemetries w/ Highest RMSE'
    olcf_low_df['Dataset'] = 'OLCF'
    olcf_low_df['Telemetries'] = 'Telemetries w/ Lowest RMSE'
    df_all = pd.concat([jlab_high_df, jlab_low_df, olcf_high_df, olcf_low_df], ignore_index=True)
    df_all['RMSE'] = df_all['RMSE'].apply(lambda x: format_e(x, mantissa_digits=1))
    df_all = df_all[['Dataset', 'Telemetries', 'features', 'model', 'RMSE']]
    df_all.set_index('Dataset', inplace=True)
    print(df_all.to_markdown()) 

if __name__=='__main__':
    args = get_args()
    args.reload_args = True

    dataset_list = ['jlab','jlab','jlab','jlab','jlab','olcfcutsec','olcfcutsec','olcfcutsec','olcfcutsec','olcfcutsec']
    model_list = ['gat','gdn','fmnm','fmtm','fm','gat','gdn','fmtm','fmnm','fm']
    load_model_paths = [
        "results/jlab_gat_03-13--18-58-08/best_03-13--18-58-08.pt",
        "results/jlab_gdn_03-13--16-01-00/best_03-13--16-01-00.pt",
        "results/jlab_fmnm_04-14--04-32-10/best_04-14--04-32-10.pt",
        "results/jlab_fmtm_03-18--02-46-32/best_03-18--02-46-32.pt",
        "results/jlab_fm_03-13--13-35-29/best_03-13--13-35-29.pt",
        "results/olcfcutsec_gat_07-08--20-18-18/best_07-08--20-18-18.pt",
        "results/olcfcutsec_gdn_05-30--01-49-01/best_05-30--01-49-01.pt",
        "results/olcfcutsec_fmtm_07-09--00-58-31/best_07-09--00-58-31.pt",
        "results/olcfcutsec_fmnm_04-14--04-55-00/best_04-14--04-55-00.pt",
        "results/olcfcutsec_fm_05-14--00-59-47/best_05-14--00-59-47.pt",
    ]

    jlab_high, jlab_low, olcf_high, olcf_low, model_order = process_data(
        rename_model_name, retrieve_old_feature_score, args, dataset_list, model_list, load_model_paths
    )
    print_results(jlab_high, jlab_low, olcf_high, olcf_low, model_order)
