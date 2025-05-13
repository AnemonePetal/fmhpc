import torch
import numpy as np
import random
import os

def set_deterministic(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True,warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

