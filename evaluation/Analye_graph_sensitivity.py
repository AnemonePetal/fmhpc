import sys
import os 
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import pandas as pd
import numpy as np
from util.extract_graph import get_similarity
from util.net_prep import convert_adj2edges,convert_edges2adj

def extract_number(s):
    return float(''.join([ch for ch in s if ch.isdigit() or ch == '.']))

def custom_resampler(group):
    num_jobs= group['allocation_id'].nunique()
    all_jobs = group['allocation_id'].value_counts(sort=True).to_json()
    if group['allocation_id'].nunique() == 1:
        flag_single_job = 1
    elif group['allocation_id'].nunique() == 0:
         flag_single_job = -1
    else:
         flag_single_job = 0
    return pd.DataFrame({'num_jobs': [num_jobs], 'all_jobs': [all_jobs], 'flag_single_job': [flag_single_job]})


def load_and_process_graph(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df = pd.DataFrame(convert_adj2edges(df), columns=['source', 'destination', 'value'])
        df['source'] = df['source'].apply(lambda x: int(x))
        df['destination'] = df['destination'].apply(lambda x: int(x))
        return df
    return None

def analyze_graph_sensitivity(global_g_path, path,title,index_name,ratio_flag=False):
    df_global = load_and_process_graph(global_g_path)
    if df_global is None:
        print(f"Error: Global graph file not found at {global_g_path}")
        return

    df_dicts = {}
    dirs = os.listdir(path)
    for dir_name in dirs:
        dir_path = os.path.join(path, dir_name)
        if os.path.isdir(dir_path):
            try:
                extract_number(dir_name)
            except ValueError:
                print(f"Skipping directory with non-numeric name: {dir_name}")
                continue

            filepath = os.path.join(dir_path, "profile_graph_adj.csv")
            df = load_and_process_graph(filepath)
            if df is not None:
                df_dicts[dir_name] = df

    param_keys = list(df_dicts.keys())
    param_keys = sorted(param_keys, key=extract_number)
    n_params = len(param_keys)

    if n_params == 0:
        print(f"No valid graph files found in subdirectories of {path}")
        return

    if ratio_flag:
        try:
            last_param_val = extract_number(param_keys[-1])
            if last_param_val == 0:
                 print(f"Error: Cannot calculate ratio, last parameter value is zero in {path}")
                 return
            param_columns = [extract_number(param) / last_param_val for param in param_keys]
            param_columns = [round(ratio, 2) for ratio in param_columns]
            param_columns = [f"{int(ratio * 100)}%" for ratio in param_columns]
        except ValueError:
             print(f"Error: Could not convert directory names to numbers for ratio calculation in {path}")
             return
        except IndexError:
             print(f"Error: No parameters found to calculate ratio in {path}")
             return
    else:
        param_columns = param_keys

    print()
    print(f'------ {title} ------')
    print()

    sens_matrix2 = np.zeros((1, n_params))
    for i in range(n_params):
        current_key = param_keys[i]
        sens_matrix2[0, i] = get_similarity(df_dicts[current_key], df_global, distance='common')

    sens_df2 = pd.DataFrame(sens_matrix2, index=[index_name], columns=param_columns)

    print(sens_df2.to_markdown())



analyze_graph_sensitivity(global_g_path ='results/jlab_fm_03-13--13-35-29/graph_datasize/72hour/profile_graph_adj.csv', path = "results/jlab_fm_03-13--13-35-29/graph_datasize", title = "Graph similarity analysis with varying durations", index_name='JLab')
analyze_graph_sensitivity(global_g_path ='results/jlab_fm_03-13--13-35-29/graph_datasize/12hour/profile_graph_adj.csv', path = "results/jlab_fm_03-13--13-35-29/graph_instances(12hour)", title = "Graph similarity analysis with varying node ratios", index_name='JLab',ratio_flag=True)
analyze_graph_sensitivity(global_g_path ='results/olcfcutsec_fm_05-14--00-59-47/graph_datasize/72hour/profile_graph_adj.csv', path = "results/olcfcutsec_fm_05-14--00-59-47/graph_datasize", title = "Graph similarity analysis with varying durations", index_name='OLCF')
analyze_graph_sensitivity(global_g_path ='results/olcfcutsec_fm_05-14--00-59-47/graph_datasize/12hour/profile_graph_adj.csv', path = "results/olcfcutsec_fm_05-14--00-59-47/graph_instances(12hour)", title = "Graph similarity analysis with varying node ratios", index_name='OLCF',ratio_flag=True)
