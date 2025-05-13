import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd

def read_result(saved_dir, saveflag='anomaly_detect'):
    result = pd.read_csv(saved_dir+'/scores/'+saveflag+'/best_eval_result.csv',index_col=0).T
    return result

def get_result(saved_model_path):
    if 'jlab' in saved_model_path:
        attack_types = ["contextual_exp_6std_cpu","cutoff_cpu","speedup_6_cpu","spikes_exp_6std_cpu","wander_cpu","contextual_exp_6std_mem","cutoff_mem","speedup_6_mem","spikes_exp_6std_mem","wander_mem","contextual_exp_6std_disk","cutoff_disk","speedup_6_disk","spikes_exp_6std_disk","wander_disk"]
    elif 'olcfcutsec' in saved_model_path:
        attack_types = ["contextual_exp_6std_cpu", "cutoff_cpu", "speedup_6_cpu", "spikes_exp_6std_cpu", "wander_cpu", "contextual_exp_6std_gpu", "cutoff_gpu", "speedup_6_gpu", "spikes_exp_6std_gpu", "wander_gpu", "contextual_exp_6std_wholepower", "cutoff_wholepower", "speedup_6_wholepower", "spikes_exp_6std_wholepower", "wander_wholepower"]
    else:
        raise ValueError("Unknown dataset in saved_model_path")

    saved_dir_path=os.path.dirname(saved_model_path)
    df_all = pd.DataFrame()
    df_instance_all = pd.DataFrame()
    for attack_type in attack_types:
        if "gat" in saved_model_path:
            df = read_result(os.path.join(saved_dir_path,attack_type), saveflag='anomaly_detect(graph)')
        elif "gdn" in saved_model_path:
            df = read_result(os.path.join(saved_dir_path,attack_type), saveflag='anomaly_detect(graph_topk)')
        else:
            df = read_result(os.path.join(saved_dir_path,attack_type), saveflag='anomaly_detect(profile_graph_topk)')
        
        df['attack_type'] = attack_type 
        df_all = pd.concat([df_all, df])

    df_all['attack_device']=df_all['attack_type'].apply(lambda x: x.split('_')[-1])
    df_all['attack_mode']=df_all['attack_type'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    df_all.to_csv(os.path.join(saved_dir_path, 'anomaly_all_result.csv'))
    mean_columns=['best_threshold_per', 'f1', 'f1_graph', 'auc_score(adjusted)']
    sum_columns=['tp', 'tp_graph', 'fp', 'fp_graph', 'fn', 'fn_graph', 'tn', 'tn_graph']
    df_view_all = df_all.groupby('attack_mode').agg({**{col: 'mean' for col in mean_columns}, **{col: 'sum' for col in sum_columns}})
    df_view_all.to_csv(os.path.join(saved_dir_path, 'anomaly_all_view.csv'))
    if "fm" in saved_model_path:
        return df_all['f1_graph'].mean(), df_all['tp_graph'].sum(), df_all['fp_graph'].sum()
    else:
        return df_all['f1'].mean(), df_all['tp'].sum(), df_all['fp'].sum()


df = pd.DataFrame()
saved_model_paths = [
    "results/jlab_gat_01-00--00-00-00/best_03-13--18-58-08.pt",
    "results/jlab_gdn_01-00--00-00-00/best_03-13--16-01-00.pt",
    "results/jlab_fm_01-00--00-00-00/best_03-13--13-35-29.pt",
]

for saved_model_path in saved_model_paths:
    if "gat" in saved_model_path:
        model = 'GAT'
    elif "gdn" in saved_model_path:
        model = "GDN"
    else:
        model = "Mantis"

    f1, tp, fp = get_result(saved_model_path)
    row = {'model': model, 'f1': round(f1,2), 'tp': tp, 'fp': fp}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

print(df.to_markdown(index=False))

saved_model_paths = [
    "results/olcfcutsec_gat_01-00--00-00-00/best_07-08--20-18-18.pt",
    "results/olcfcutsec_gdn_01-00--00-00-00/best_05-30--01-49-01.pt",
    "results/olcfcutsec_fm_01-00--00-00-00/best_05-14--00-59-47.pt",
]
df = pd.DataFrame()

for saved_model_path in saved_model_paths:
    if "gat" in saved_model_path:
        model = 'GAT'
    elif "gdn" in saved_model_path:
        model = "GDN"
    else:
        model = "Mantis"

    f1, tp, fp = get_result(saved_model_path)
    row = {'model': model, 'f1': round(f1,2), 'tp': tp, 'fp': fp}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

print(df.to_markdown(index=False))
