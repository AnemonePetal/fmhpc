# Installation
## Prepare environment

### Method 1. Auto install script
```bash
./autosetup.sh
```
### Method 2. Manually Install Packages



```bash
conda env create --file environment.yml # init conda environment
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
pip install torch-geometric==1.5.0
```

```bash
pip install git+https://github.com/epfl-lts2/pygsp
```


Modify library files to be compatible
Modify the file `dataloader.py` in the `torch_geometric` library under your conda env folder. (It may appear under `/home/user/miniconda3/envs/fmhpc/lib/python3.8/site-packages/torch_geometric/data/dataloader.py`)

Replace line 5
```python
from torch._six import container_abcs, string_classes, int_classes
```
with the following codes:
```python
import collections.abc as container_abcs
int_classes = int
string_classes = str
```

### Prepare environment for Prodigy
```
conda env create --file ./build_env/env_vae.yml
```


If you run into any problems, refer to the guidance in the Prodigy [repository](https://github.com/peaclab/Prodigy).

