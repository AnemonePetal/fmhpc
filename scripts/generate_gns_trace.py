import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
from models.GDN import GDN_wrapper
from models.GAT import GAT_wrapper
from models.FM_TM import FM_TM_wrapper
from models.FM import FM_wrapper



def main():
    args = get_args()
    args.dataset = 'olcfcutsec'
    args.model = 'fm'
    args.load_model_path = 'results/olcfcutsec_fm_05-14--00-59-47/best_05-14--00-59-47.pt'
    args.save_subdir = 'fm'
    args.batch = 1024
    args.deterministic = True
    args.epoch = 60
    args.device = 'cuda:0'
    prepare_env(args)
    args.test_file = '202001_norm_test.parquet'
    data = Dataset(args)
    model = FM_wrapper(args,data,tensorboard=True)
    if model.load():
        raise Exception('Not Trained Model')
    model.evolve_graph_sim_eval()

if __name__=="__main__":
    main()