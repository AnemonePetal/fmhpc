import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
from models.FM import FM_wrapper



def main():
    print('Compute TRGs construction cost with varying data sizes')
    print('> Compute TRGs construction cost on JLAB')
    args = get_args()
    args.dataset = 'jlab'
    args.model = 'fm'
    args.load_model_path = 'results/jlab_fm_03-13--13-35-29/best_03-13--13-35-29.pt'
    args.save_subdir = 'fm'
    args.batch = 32
    args.deterministic = True
    args.epoch = 100
    args.device = 'cuda:0'
    prepare_env(args)
    data = Dataset(args)
    model = FM_wrapper(args,data,tensorboard=True)
    if model.load():
        raise Exception('Not Trained Model')
    model.measure_graph_time()

    print('> Compute TRGs construction cost on OLCF')
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
    model.measure_graph_time()

if __name__=="__main__":
    main()