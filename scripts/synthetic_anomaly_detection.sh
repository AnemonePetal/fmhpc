#!/bin/bash


attack_types=("contextual_exp_6std_cpu" "cutoff_cpu" "speedup_6_cpu" "spikes_exp_6std_cpu" "wander_cpu" "contextual_exp_6std_mem" "cutoff_mem" "speedup_6_mem" "spikes_exp_6std_mem" "wander_mem" "contextual_exp_6std_disk" "cutoff_disk" "speedup_6_disk" "spikes_exp_6std_disk" "wander_disk")
saved_model_path="results/jlab_fm_01-00--00-00-00/best_03-13--13-35-29.pt"
for attack_type in "${attack_types[@]}"
do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main.py -model fm -threshold_per -1 --no_pCache --attack --retain_beforeattack -load_model_path $saved_model_path -dataset jlab -save_subdir $attack_type -dtask anomaly --deterministic -slide_win 10 -slide_stride 1 -batch 32 -epoch 100 -early_stop_win 100 -random_seed 5 -dim 64 -out_layer_num 3 -out_layer_inter_dim 128 -scaler minmax
done


attack_types=("contextual_exp_6std_cpu" "cutoff_cpu" "speedup_6_cpu" "spikes_exp_6std_cpu" "wander_cpu" "contextual_exp_6std_gpu" "cutoff_gpu" "speedup_6_gpu" "spikes_exp_6std_gpu" "wander_gpu" "contextual_exp_6std_wholepower" "cutoff_wholepower" "speedup_6_wholepower" "spikes_exp_6std_wholepower" "wander_wholepower")

saved_model_path="results/olcfcutsec_fm_01-00--00-00-00/best_05-14--00-59-47.pt"
for attack_type in "${attack_types[@]}"
do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main.py -model fm -threshold_per -1 --no_pCache --attack -load_model_path $saved_model_path -dtask anomaly -dataset olcfcutsec -save_subdir $attack_type  --deterministic -slide_win 10 -slide_stride 1 -batch 1024 -epoch 60 -early_stop_win 100  -random_seed 5 -dim 64 -out_layer_num 3 -out_layer_inter_dim 128 -scaler minmax -device cuda:0
done