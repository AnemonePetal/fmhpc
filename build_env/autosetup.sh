#!/bin/bash

# Get the path to the base environment
conda_path=$(conda info --base 2>&1)
if [ $? -ne 0 ]; then
    echo "Error: Conda is not installed or not in the PATH"
    exit 1
fi

envs_dir="$conda_path/envs"
env_names=$(ls -1 "$envs_dir" | grep -v "^__")
env_name="fmhpc"

if echo "$env_names" | grep -q "$env_name"; then
    echo "environment already exists"
    exit 0
else
    echo "Creating environment"
    env_file="./build_env/fmhpc.yml"
    conda env create -f "$env_file"
    source "$conda_path/bin/activate" "$env_name"
    conda env list
    pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
    pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
    pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
    pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
    pip install torch-geometric==1.5.0
    echo "Modify ataloader.py in torch_geometric"
    dataloader_file="$envs_dir/$env_name/lib/python3.8/site-packages/torch_geometric/data/dataloader.py"
    if grep -q "from torch._six import container_abcs, string_classes, int_classes" "$dataloader_file"; then
        sed -i '5,6s/from torch._six import container_abcs, string_classes, int_classes/import collections.abc as container_abcs\nint_classes = int\nstring_classes = str/' "$dataloader_file"
    fi
fi