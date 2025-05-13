# Quick Reproduce
- Table2
```
conda activate fmhpc
python evaluation/Regression_score.py
```

- Table3
```
conda activate fmhpc
python evaluation/Profile_model.py
```

- Fig 5,6
```
conda activate fmhpc
python evaluation/Plot_graph.py
```

- Fig 7
```
conda activate fmhpc
python evaluation/Compare_line.py
```

- Fig. 8
```
conda activate fmhpc
python scripts/generate_gns_trace.py # optional
python evaluation/Plot_gns.py
```

- Table 4
```
conda activate fmhpc
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python evaluation/Timing_graph_cost.py
```

- Table 5
```
conda activate fmhpc
python evaluation/Analye_graph_sensitivity.py
```

- Table 6,7
```
conda activate fmhpc
python evaluation/Eval_synthetic_anomalies.py
```

- Table 8
```
conda activate fmhpc
python evaluation/Eval_sandia_anomalies.py
```

- Table 9
```
conda activate fmhpc
python evaluation/Eval_olcf_realworld_anomalies.py
```

- Figure 9
```
conda activate fmhpc
python evaluation/Plot_anomaly_time_win.py
```

# Start from scratch
The following commands show how to train models on JLab. To train models using the OCLF dataset instead, please change the `-dataset jlab` option to `-dataset olcfcutsec`.

```
conda activate fmhpc

# Train FM model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main.py --no_pCache -dataset jlab -save_subdir fm -model fm --deterministic -batch 32 -epoch 100

# Train GDN model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main.py --no_pCache -dataset jlab -save_subdir gdn -model gdn --deterministic -batch 32 -epoch 100

# Train GAT model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main.py --no_pCache -dataset jlab -save_subdir gat -model gat --deterministic -batch 32 -epoch 100

# Train Prodigy model
conda activate vae
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main_vae.py --no_pCache -dataset jlab -save_subdir vae -model vae --deterministic -batch 32 -epoch 100
```

The following scripts generate results from trained models.
- Synthetic Anomalies
```
conda activate fmhpc
# run FM model
./scripts/synthetic_anomaly_detection.sh

# run GAT model
./scripts/baseline/synthetic_anomaly_detection_gat.sh

# run GDN model
./scripts/baseline/synthetic_anomaly_detection_gdn.sh


python scripts/merge_synthetic_anomaly_result.py
```

- OLCF XID GPU Anomalies
```
conda activate fmhpc
# run FM model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/olcf_xid_workflow_bytol.py
# run GAT model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/baseline/olcf_xid_workflow_bytol_gat.py

# run GDN model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/baseline/olcf_xid_workflow_bytol_gdn.py

# run Prodigy model
conda activate vae
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/baseline/olcf_xid_workflow_bytol_vae.py
```

# Sensitiviy analysis on TRG epsilon
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/analysis_graph_epsilon.py
```