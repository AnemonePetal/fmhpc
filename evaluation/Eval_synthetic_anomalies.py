import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np

def convert_decimal_to_int(decimal_num):
  s_num = str(decimal_num)
  if s_num.startswith("0.") and len(s_num) > 2 and s_num[2:].isdigit():
    integer_part_str = s_num[2:]
    return int(integer_part_str)
  else:
    print(f"Warning: Input {decimal_num} is not in the expected '0.digits' format for this specific conversion.")
    return None

def read_result(saved_dir, threshold=0.999):
    threshold = convert_decimal_to_int(threshold)
    result = pd.read_csv(saved_dir+f'/anomaly_all_result_athr{threshold}.csv')
    return result

def get_result(saved_model_path,threshold):

    saved_dir_path=os.path.dirname(saved_model_path)
    df = read_result(saved_dir_path,threshold)
    if "fm" in saved_dir_path:
        return df['f1_graph'].mean(), df['tp_graph'].sum(), df['fp_graph'].sum()
    else:
        return df['f1'].mean(), df['tp'].sum(), df['fp'].sum()


saved_model_paths = [
    "results/jlab_gat_01-00--00-00-00/best_03-13--18-58-08.pt",
    "results/jlab_gdn_01-00--00-00-00/best_03-13--16-01-00.pt",
    "results/jlab_fm_01-00--00-00-00/best_03-13--13-35-29.pt",
]
thresholds = [0.9999,0.999]
print('--- JLab Synthetic Anomalies Dataset ---')
for threshold in thresholds:
    print(f'> Anomaly Score> {threshold*100}% of train anomaly scores')
    df = pd.DataFrame()
    for saved_model_path in saved_model_paths:
        if "gat" in saved_model_path:
            model = 'GAT'
        elif "gdn" in saved_model_path:
            model = "GDN"
        else:
            model = "Mantis"

        f1, tp, fp = get_result(saved_model_path,threshold)
        row = {'model': model, 'f1': round(f1,2), 'tp': tp, 'fp': fp}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    print(df.to_markdown(index=False))

saved_model_paths = [
    "results/olcfcutsec_gat_01-00--00-00-00/best_07-08--20-18-18.pt",
    "results/olcfcutsec_gdn_01-00--00-00-00/best_05-30--01-49-01.pt",
    "results/olcfcutsec_fm_01-00--00-00-00/best_05-14--00-59-47.pt",
]
print()
print('--- OLCF Synthetic Anomalies Dataset ---')
for threshold in thresholds:
    print(f'> Anomaly Score> {threshold*100}% of train anomaly scores')
    df = pd.DataFrame()
    for saved_model_path in saved_model_paths:
        if "gat" in saved_model_path:
            model = 'GAT'
        elif "gdn" in saved_model_path:
            model = "GDN"
        else:
            model = "Mantis"

        f1, tp, fp = get_result(saved_model_path,threshold)
        row = {'model': model, 'f1': round(f1,2), 'tp': tp, 'fp': fp}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    print(df.to_markdown(index=False))
