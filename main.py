import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
from models.GDN import GDN_wrapper
from models.GAT import GAT_wrapper
from models.FM_TM import FM_TM_wrapper
from models.FM import FM_wrapper
from models.FM_NM import FM_NM_wrapper

def main():
    args = get_args()
    prepare_env(args)
    data = Dataset(args)
    
    
    if args.model == 'gdn':
        model = GDN_wrapper(args,data,tensorboard=True)
    elif args.model == 'gat':
        model = GAT_wrapper(args,data,tensorboard=True)
    elif args.model == 'fmtm':
        model = FM_TM_wrapper(args,data,tensorboard=True)
    elif args.model == 'fm':
        model = FM_wrapper(args,data,tensorboard=True)
    elif args.model == 'fmnm':
        args.hnodes_size = data.train.instance.nunique()
        model = FM_NM_wrapper(args,data,tensorboard=True)
    else:
        raise Exception('{} model not implemented'.format(args.model))
    
    if model.load():
        model.train()
    model.score()


if __name__=="__main__":
    main()