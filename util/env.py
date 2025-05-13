import numpy as np
import os
from util.save import get_save_path, update_savepath_datestr
import json5

_device = None 

def get_device():
    return _device

def set_device(dev):
    global _device
    _device = dev

def init_work(worker_id, seed):
    np.random.seed(seed + worker_id)

def load_old_args(args):
    if not os.path.exists(os.path.dirname(args.paths['test_re'])+'/args.json'):
        print('args.json not found, use provided args in cli')
        return 0
    skip_keys = ['load_model_path']
    with open(os.path.dirname(args.paths['test_re'])+'/args.json') as f:
        data_args = json5.load(f)
        for key in data_args:
            if key not in skip_keys:
                setattr(args, key, data_args[key])
    print("args loaded from file:"+ os.path.dirname(args.paths['test_re'])+'/args.json')
    return 0

def human_readable_args(args):
    if args.no_pCache == True:
        args.pCache = False
    else:
        args.pCache = True
    
    if args.no_store == True:
        args.store = False
    else:
        args.store = True



def prepare_env(args):
    if hasattr(args,'early_stop_win') and args.early_stop_win == -1:
        args.early_stop_win = args.epoch
    update_savepath_datestr(args)
    get_save_path(args,build_dir=not args.no_store)
    if hasattr(args, 'load_model_path') and  args.load_model_path != '' and args.reload_args:
        load_old_args(args)
    set_device(args.device)
    human_readable_args(args)

