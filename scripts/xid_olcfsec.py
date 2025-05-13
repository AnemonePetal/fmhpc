import sys
import os
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import copy

from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
from models.GDN import GDN_wrapper
from models.GAT import GAT_wrapper
from models.FM_TM import FM_TM_wrapper
from models.FM import FM_wrapper



def main_xids(tol_min=None, xids = [13,31,43,45,48,61,62,63,64,74,79], set_args=None):
    if tol_min is None:
        raise Exception('tol_min cannot be None')
        
    args_init = get_args()
    if set_args is not None:
        for key in set_args:
            setattr(args_init, key, set_args[key])                            
    if hasattr(args_init,'tol_min'):
        raise Exception('args_init should not have tol_min attribute')

    args_init.tol_min = tol_min

    for xid in xids:
        args = copy.deepcopy(args_init)

        args.test_file = f'{xid}_test_'+str(tol_min)+'mintol.parquet'
        suffix = args.test_file.split('_')[-1].split('.')[0]
        args.test_ignoresync = True
        args.save_subdir='xid'+str(xid)+'/'+suffix

        prepare_env(args)
        data = Dataset(args)
        
        print(f'Running model for {args.test_file}')

        if args.model == 'gdn':
            model = GDN_wrapper(args,data,tensorboard=True)
        elif args.model == 'gat':
            model = GAT_wrapper(args,data,tensorboard=True)
        elif args.model == 'fmtm':
            model = FM_TM_wrapper(args,data,tensorboard=True)
        elif args.model == 'fm':
            model = FM_wrapper(args,data,tensorboard=True)

        else:
            raise Exception('{} model not implemented'.format(args.model))
        if model.load():
            model.train()
        model.score()



if __name__=="__main__":
    main_xids()
