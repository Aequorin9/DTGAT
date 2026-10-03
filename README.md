# DTGAT

DTGAT is a dual-task graph attention model for gene regulatory network (GRN)
inference. The model jointly performs TF-target link prediction and gene
expression reconstruction.

This repository contains the Arabidopsis implementation and the processed
input files used for model evaluation.

## Repository structure

```text
DTGAT/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── DTGAT_Arabidopsis.py   # executable training entry point
│   ├── dtgat_config.py        # device, seed, and shared constants
│   ├── dtgat_layers.py        # sparse graph-attention core
│   ├── dtgat_model.py         # DTGAT architecture and early stopping
│   ├── data_utils.py          # data loading, masking, and graph construction
│   ├── training_utils.py      # dual-task training loop
│   ├── evaluation_utils.py    # final evaluation and output serialization
│   ├── plotting_utils.py      # consolidated training figure
│   └── runtime_utils.py       # logging and environment reporting
└── data/
    └── Arabidopsis/
        ├── expression_matrix.csv
        ├── reference_network.csv
        └── README.md
```

The script writes the trained model, network predictions, summary statistics,
a consolidated training figure, and a timestamped log to an `output/`
directory created automatically in the repository root.

The implementation is organized by responsibility. The entry-point script
contains only the end-to-end workflow, while the graph-attention layer, DTGAT
architecture, data pipeline, training, evaluation, plotting, and runtime
utilities are maintained in separate modules under `src/`.

## Installation

Python 3.9 or later is recommended. Create an isolated environment and install
the required packages:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build that matches the local CUDA
version before installing the remaining dependencies.

## Input data

The script reads the following files using paths relative to the repository:

- `data/Arabidopsis/expression_matrix.csv`
- `data/Arabidopsis/reference_network.csv`

Their formats are described in
[`data/Arabidopsis/README.md`](data/Arabidopsis/README.md).

## Running DTGAT

From the repository root, run:

```bash
python src/DTGAT_Arabidopsis.py
```

The default seed is 42. The script performs the 60%/20%/20% split of positive
regulatory relationships, samples negative TF-target pairs, trains DTGAT,
selects the best checkpoint using validation AUPR (with validation AUC as the
secondary criterion), and reports performance on the held-out test set.
The first five epochs are printed to the terminal; subsequent progress is
printed every 50 epochs. The timestamped log retains the complete run record.

## Main outputs

The automatically created `output/` directory includes:

- `DTGAT.pth`: trained model and associated metadata;
- `predicted_network_all.csv`: scores for all evaluated TF-target candidates;
- `prediction_statistics.csv`: counts of training-prior, held-out-prior, and
  novel candidate relationships;
- `reconstructed_expression.csv`: expression values reconstructed by the
  expression-reconstruction task;
- `dual_task_training_curves.png`: consolidated training and validation curves;
- `training_YYYYMMDD_HHMMSS.log`: runtime settings, progress, and evaluation
  metrics.

Runtime outputs are written to `output/`, which is ignored by Git. Curated
manuscript figures, if included in a release, should be placed in a separate
`figures/` directory.

## Implementation provenance

DTGAT extends the graph-attention encoder design used by
[GENELink](https://github.com/zpliulab/GENELink). The adapted implementation in
`src/dtgat_layers.py` replaces dense all-pairs attention with sparse edge-wise
attention and adds residual connections. DTGAT further introduces a third
attention layer, joint TF-target link prediction and expression-reconstruction
objectives, and its associated training and evaluation workflow.

Please cite both the associated DTGAT manuscript and the original GENELink
publication when using this code.

## License

The DTGAT repository is released under the MIT License. See [`LICENSE`](LICENSE).

## Data availability

The processed Arabidopsis expression matrix and reference regulatory network
used by this implementation are included under `data/Arabidopsis/`. Details of
the original data sources and processing procedures are provided in the
associated manuscript.

