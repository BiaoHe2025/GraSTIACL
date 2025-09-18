# GraSTIACL

GraSTIACL is a graph-based deep learning framework for brain-network analysis. It integrates structural and functional connectivity with an information-bottleneck module for robust and interpretable classification. This repository supports experiments using ADNI and SRPBS (MDD) public datasets and an internal private dataset (BD).
<img src="image.png" alt="替代文字" width="400"/>

---

## Requirements

This project targets Python 3.9+ and CUDA 12.1. Install the required packages (example pip wheels shown):

```
torch==2.5.0+cu121
torch-cluster==1.6.3+pt25cu121
torch-geometric==2.6.1
torch-scatter==2.1.2+pt25cu121
torch-sparse==0.6.18+pt25cu121
torch-spline-conv==1.2.2+pt25cu121
torchaudio==2.5.0+cu121
torchvision==0.20.0+cu121
pyg-lib==0.4.0+pt25cu121
```

Install (example):

```bash
pip install torch==2.5.0+cu121 torchaudio==2.5.0+cu121 torchvision==0.20.0+cu121 \
  torch-scatter==2.1.2+pt25cu121 torch-sparse==0.6.18+pt25cu121 \
  torch-cluster==1.6.3+pt25cu121 torch-spline-conv==1.2.2+pt25cu121 \
  torch-geometric==2.6.1 pyg-lib==0.4.0+pt25cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

> Note: Use the appropriate wheel index for your CUDA/PyTorch combination if different from CUDA 12.1.

---

## Data format and organization

Preprocessing is performed in MATLAB. After preprocessing, place the processed data under `data/` using the following layout (take AD group as an example and each item stores files for all subjects of the group):

```
data/
├── AD_ADJ/    # AD group adjacency matrices
├── AD_DW/     # AD group dynamic edge-weight matrices
├── AD_NF/     # AD group node-feature matrices
├── NC_ADJ/    # NC group adjacency matrices
├── NC_DW/     # NC group dynamic edge-weight matrices
├── NC_NF/     # NC group node-feature matrices

```

Each subject should have corresponding `.mat` or similarly-loadable matrices named consistently (e.g., `<subj_id>_adj.mat`, `<subj_id>_dw.mat`, `<subj_id>_nf.mat`). All matrices for a subject must share compatible dimensions so graphs can be constructed reliably.

**Datasets used**

- **ADNI**: public Alzheimer’s Disease dataset.
- **SRPBS (MDD)**: public MDD subset from SRPBS.
- **BD**: private dataset used in internal experiments — treat as restricted/private.

---

## How to run

The repository contains one main scripts:

- `GraSTIACL.py` — training and evaluation pipeline. Loads processed data, constructs graphs, trains the model, and logs metrics.

Run training with:

```bash
python GraSTIACL.py
```

Example arguments (implementations may vary; check `GraSTIACL.py` for supported CLI flags):

- `--root` : path to the `data/` directory.
- `--dataset` : one of `ADNI`, `SRPBS`, `BD`, or `ALL` (if supported).
- `--epochs` : number of epochs.
- `--batch_size` : training batch size.
- `--device` : `cuda` or `cpu`.

---

## Project structure

```
├── GraSTIACL.py        # Main training pipeline (entry point)
├── unsupervised/       # Additional model components
├── datasets/           # Load processed datasets (see Data format)
└── README.md           # This file
```

---

## Citation

If you use GraSTIACL in your research, please cite
```bibtex
@article{He2025GraSTIACL,
  author  = {He, Biao and Ji, Erni and Zong, Xiaofen and Liang, Zhen and Huang, Gan and Zhang, Li},
  title   = {GraSTI-ACL: Graph Spatial-Temporal Infomax with Adversarial Contrastive Learning for Brain Disorders Diagnosis Based on Resting-State fMRI},
  journal = {Medical Image Analysis},
  year    = {2025}
}
```

---

## Contact

For questions about the code, open an issue or contact the repository owner.

