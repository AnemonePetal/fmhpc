# Mantis

<div align="center">
  <img src="docs/overview.png" alt="Mantis Overview" width="600">
</div>

Mantis is a framework for predicting and interpreting High-Performance Computing (HPC) telemetry data. It performs predictive modeling on unlabeled data to capture both the temporal relationships within individual telemetry streams and the complex interactions between different streams.

# Install

Install the Mantis environment using conda. We provide a script for quick installation:

```bash
./build_env/autosetup.sh
```

If you run into any issues with the script, you can set up the environment manually according to the steps in `build_env/README.md`.

# Dataset

All data is publicly available.

| Dataset Name | # of telemetry | # of hosts | size (GB) | data source                                       | data collect frequency                | Download |
| ------------ | -------------- | ---------  | --------- | ------------------------------------------------- | ------------------------------------- | -------- |
| JLab         | 66             | 332        | 180       | Prometheus, Slurm                                 | 1 min per job                         | [1]      |
| OLCF         | 28             | 4626       | 492       | GPULog, OpenBMC, Job scheduler allocation history | at occurrence; 1 sec / 10 sec per job | [2,3]    |



[1] [Dataset for Investigating Anomalies in Compute Clusters](https://zenodo.org/records/10058230  )

[2] [Long Term Per-Component Power and Thermal Measurements of the OLCF Summit System](https://doi.ccs.ornl.gov/dataset/086578e9-8a9f-56b1-a657-0ed8b7393deb)  

[3] [OLCF Summit Supercomputer GPU Snapshots During Double-Bit Errors and Normal Operations](https://doi.ccs.ornl.gov/dataset/56c244d2-d273-5222-8f4b-f2324282fab8)

 Perfect synchronization is impossible during extremely large-scale data collection from hundreds of nodes with varied hardware telemetry. Therefore, the raw data suffers from unstable sampling rates and missing values. We cleaned the data using `PySpark` and `Dask` and will release the final dataset for training, validation, and testing.

# Example

## Quick Start
Start Mantis with the script:
```
conda activate fmhpc
./scripts/run.sh
```
The output results will be stored in the `result` folder.

## Visualize result

### Telemetry Analysis
Mantis enables accurate prediction of HPC telemetry streams. Using `vizro` and `plotly`, Mantis provides interactive charts for detailed analysis, as shown by the successful predictions in the example below. The chart allows you toquickly inspect each telemetry on any compute nodes.

```
python visualization/monitor_page.py --no_pCache -threshold_per 0.9999 -load_model_path results/may_tsmixer_03-13--13-35-29/best_03-13--13-35-29.pt -dtask anomaly -dataset may -save_subdir tsmixer -model tsmixer --deterministic -slide_win 10 -graph_skip_conn 0.15 -edge_thr 1e-10 -topk 66 -slide_stride 1 -batch 32 -epoch 100 -early_stop_win 100 -comment "" -random_seed 5 -decay 0 -dim 64 -out_layer_num 3 -out_layer_inter_dim 128 -scaler minmax 
```
<div align="center">
  <img src="docs/example1.png" alt="Telemetry Analysis Example" width="800">
</div>

### Telemetry Relation Graph (TRG)
Mantis enables interpretation of telemetry relationships with TRGs. With the help of `plotly`, Mantis provides interactive 3d network charts to visualize TRGs.
```
python visualization/interactice_network.py
```
<div align="center">
  <img src="docs/example2.png" alt="Telemetry Relation Graph Example" width="800">
</div>

With the help of `networkx`, Mantis could further reveal the core structures of the complex TRGs.
```
python visualization/core_network.py
```
<div align="center">
  <img src="docs/example3.png" alt="Telemetry Relation Graph Example" width="600">
</div>


# Hyper-parameter Analysis

## Model Architecture Search
To understand how hyper-parameters influence the model quality,

## Telemetry Relation Graph (TRG)
testsdfdsf