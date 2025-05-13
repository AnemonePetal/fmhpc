import argparse
import re
import json
import os

def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument('-save_subdir', help='save subdir', type = str, default='')
        parser.add_argument('-dataset', help='jlab / olcf/ prodigy_mini', type = str, default='jlab')
        parser.add_argument('-model', help='gdn/ gat / fm / fm_nm / fm_tm /vae', type = str, default='fm')
        parser.add_argument('-dtask', help='anomaly / none', type = str, default='none')
        parser.add_argument('--attack', help='whether to use attack to build new test, val dataset', action='store_true', default=False)
        parser.add_argument('--retain_beforeattack', help='whether to retain the data before attack', action='store_true', default=False)
        parser.add_argument('--reload_args', help='whether to reload args from file', action='store_true', default=False)
        parser.add_argument('--deterministic',help='whether to use deterministic mode', action='store_true', default=False)
        parser.add_argument('--no_store', help='whether to store the result', action='store_true', default=False)
        parser.add_argument('--no_pCache', help='whether to use cache', action='store_true', default=False)
        parser.add_argument('-batch', help='batch size', type = int, default=128)
        parser.add_argument('-epoch', help='train epoch', type = int, default=100)
        parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
        parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

        parser.add_argument('-slide_win', help='slide_win', type = int, default=10)
        parser.add_argument('-slide_stride', help='slide_stride', type = int, default=1)
        parser.add_argument('-pred_win', help='pred_win', type = int, default=1)
        parser.add_argument('-scaler', help='minmax / robust', type = str, default='minmax')
        parser.add_argument('--optuna_prune', help='whether to use optuna prune', action='store_true', default=False)
        parser.add_argument('-learning_rate', help='learning rate of the optimization algorithm', type=float, default=0.001)
        parser.add_argument('-momentum', help='momentum for the optimizer', type=float, default=0.9)
        parser.add_argument('-eval_tol', help='Anomaly detection evaluation: allowed time tolerance in minutes. Default: 0 (perfect detection).', type=int, default=0)
        parser.add_argument('-threshold_per', help='Percentile threshold for anomaly detection. Default: -1 (use the best threshold found during validation).', type=float, default=-1)
        parser.add_argument('-micro_batch',help='micro batch size', type = int, default=4)
        parser.add_argument('-dataloader_num_workers',help='num_workers in the dataloader', type = int, default=16)
        parser.add_argument('-dataloader_prefetch_factor',help='prefetch_factor in the dataloader', type = int, default=4)
        parser.add_argument('--trg_epsilon',help='trg_epsilon', type = float, default=1e-2)
        parser.add_argument('--simulate_perfect_model', help='Use perfect model for perfect predictions. Only for proof of concept on pipeline.', action='store_true', default=False)
        parser.add_argument('-out_layer_num', help='model architecture parameters', type = int, default=3)
        parser.add_argument('-out_layer_inter_dim', help='model architecture parameters', type = int, default=128)
        parser.add_argument('-dim', help='model architecture parameters', type = int, default=64)
        parser.add_argument('-early_stop_win', help='early_stop_win', type = int, default=-1)
        parser.add_argument('-random_seed', help='random seed', type = int, default=5)
        args = parser.parse_args()
        return args
    
def prefix_to_regex(args):
    m_savepath =re.match("farm\d{2}",args.save_subdir)
    if m_savepath:
        args.data_filter= {
            "instance":'/'+m_savepath.group()+'.*/',
        }
    else:
        args.data_filter= {
        }
    



def get_args():
    args = parse()
    prefix_to_regex(args)
    return args
    
def save_args(args):
    args_dict = vars(args)
    if 'hnodes' in args_dict:
        args_dict.pop('hnodes')
    if 'constant_feature_mask' in args_dict:
        args_dict.pop('constant_feature_mask')
    with open(os.path.dirname(args.paths['test_re'])+'/args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)
    return 0