# Pore VNs

## Install

Main Requirements are as follow:

- torch
- torch_geometric
- ase, pymatgen
- jmp, if used
- lightning
- einops
- click

To install, please run

```sh
pip install torch==2.5.0
pip install lightning==2.5.0
pip install einops ase pymatgen colorama click scikit-learn transformers==4.49.0 python-box tensorboard tensorboardX
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install torch_geometric
```

## Data preprocessing

Please provide a folder containing `cif` folder and `benchmark.csv` (for label). An example is provided in `data/test`.

Then run

```sh
./scripts/build_graph.sh ./configs/test/n2.yaml
```

## Train

run

```sh
./scripts/train.sh ./configs/test/n2.yaml
```
