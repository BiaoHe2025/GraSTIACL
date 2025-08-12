#GraSTIACL

GraSTIACL is a graph-based deep learning framework for brain-network analysis. It integrates structural and functional connectivity with an information-bottleneck module for robust and interpretable classification. This repository supports experiments using ADNI and SRPBS (MDD) public datasets and an internal private dataset (BD).

##Requirements

This project targets Python 3.9+ and CUDA 12.1. Install the required packages (example pip wheels shown):

torch==2.5.0+cu121
torch-cluster==1.6.3+pt25cu121
torch-geometric==2.6.1
torch-scatter==2.1.2+pt25cu121
torch-sparse==0.6.18+pt25cu121
torch-spline-conv==1.2.2+pt25cu121
torchaudio==2.5.0+cu121
torchvision==0.20.0+cu121
pyg-lib==0.4.0+pt25cu121

Install (example):

pip install torch==2.5.0+cu121 torchaudio==2.5.0+cu121 torchvision==0.20.0+cu121 \
  torch-scatter==2.1.2+pt25cu121 torch-sparse==0.6.18+pt25cu121 \
  torch-cluster==1.6.3+pt25cu121 torch-spline-conv==1.2.2+pt25cu121 \
  torch-geometric==2.6.1 pyg-lib==0.4.0+pt25cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121

Note: Use the appropriate wheel index for your CUDA/PyTorch combination if different from CUDA 12.1.

Data format and organization

Preprocessing is performed in MATLAB. After preprocessing, place the processed data under data/ using the following layout (each item stores files for all subjects of the group):

data/
├── Data-AD_ADJ/    # AD group adjacency matrices
├── Data-AD_DW/     # AD group dynamic edge-weight matrices
├── Data-AD_NF/     # AD group node-feature matrices
├── Data-NC_ADJ/    # NC group adjacency matrices
├── Data-NC_DW/     # NC group dynamic edge-weight matrices
├── Data-NC_NF/     # NC group node-feature matrices
├── Data-SRPBS_ADJ/ # SRPBS (MDD) adjacency matrices
├── Data-SRPBS_DW/  # SRPBS dynamic edge-weight matrices
├── Data-SRPBS_NF/  # SRPBS node-feature matrices
├── Data-BD_ADJ/    # BD (private) adjacency matrices
├── Data-BD_DW/     # BD (private) dynamic edge-weight matrices
└── Data-BD_NF/     # BD (private) node-feature matrices

Each subject should have corresponding .npy or similarly-loadable matrices named consistently (e.g., <subj_id>_adj.npy, <subj_id>_dw.npy, <subj_id>_nf.npy). All matrices for a subject must share compatible dimensions so graphs can be constructed reliably.

Datasets used

ADNI: public Alzheimer’s Disease dataset.

SRPBS (MDD): public MDD subset from SRPBS.

BD: private dataset used in internal experiments — treat as restricted/private.

How to run

The repository contains two main scripts:

GraSTIACL.py — training and evaluation pipeline. Loads processed data, constructs graphs, trains the model, and logs metrics.

GraSTI.py — core model and information-bottleneck implementation.

Run training with:

python GraSTIACL.py --data_root ./data --dataset ADNI --epochs 200 --batch_size 16

Example arguments (implementations may vary; check GraSTIACL.py for supported CLI flags):

--data_root : path to the data/ directory.

--dataset : one of ADNI, SRPBS, BD, or ALL (if supported).

--epochs : number of epochs.

--batch_size : training batch size.

--device : cuda or cpu.

Project structure

├── GraSTIACL.py        # Main training pipeline (entry point)
├── GraSTI.py           # Model + information bottleneck
├── models/             # Additional model components
├── utils/              # Data loaders and helper utilities
├── data/               # Processed datasets (see Data format)
├── experiments/        # Training logs and checkpoints
└── README.md           # This file

Notes on privacy and data usage

The BD dataset is private. Do not share BD contents or include BD raw data in public commits. Use .gitignore to exclude private files and store them in secure storage.

Example .gitignore entries:

/data/Data-BD_*/
/experiments/checkpoints_bd/

Citation

If you use GraSTIACL in your research, please cite this repository and the original datasets (ADNI and SRPBS).

License

This project is released under the MIT License. See LICENSE for details.

Contact

For questions about the code, open an issue or contact the repository owner.
