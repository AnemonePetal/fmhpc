import sys
import os
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from scripts.xid_olcfsec_vae import main_xids

tol_mins = [1,2,3,4,5,6,7,8,9,10]

xid_list = [63,74,79]

set_args = {}
set_args['dataset'] = 'olcfcutsec'
set_args['model'] = 'vae'
set_args['load_model_path'] = 'results/olcfcutsec_vae_12-26--19-54-28/model-weights.h5'
set_args['save_subdir'] = 'vae'
set_args['deterministic'] = True
set_args['dtask'] = 'anomaly'
set_args['batch'] = 32
set_args['device'] = 'cuda:0'


for tol_min in tol_mins:
    main_xids(tol_min=tol_min,xids = xid_list, set_args = set_args)
