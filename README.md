# VAmodel

VAmodel is an experimental Python project for exploring reverse stochastic polynomials to derive value-area signals (value areas) in trading (specifically equity futures). The pipeline covers data preprocessing, model training, and inference, demonstrating a full end-to-end workflow. While this method has not yet produced a profitable edge, the code and structure showcase my approach to using transformer based architecture to identify user marked patterns.

> **Note:** Model checkpoints and actual market data are excluded. Additionally, proprietary marker data (`dev/VA_markers_final.xlsx`) and the detailed preprocessing scripts (`dev/preprocess/valmark/`) are referenced but **excluded** via `.gitignore`. A dummy template (`dev/markers_example.xlsx`) is included to illustrate the expected format.

## Table of Contents

* [Directory Structure](#directory-structure)
* [.gitignore](#gitignore)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Data Preparation](#data-preparation)
* [Training](#training)
* [Inference](#inference)
* [Development](#development)
* [Contributing](#contributing)
* [License](#license)

## Directory Structure

```
VAmodel/
├── data/
│   ├── data_edit/         # Raw market data (.parquet files)
│   ├── data_training/     # Preprocessed data for training
│   ├── data_verdict/      # Inference results and analysis
│   └── debug/             # Debug logs and temporary files
├── dev/
│   ├── config.yaml             # Configuration for training and inference
│   ├── run_training.py         # Training entry point
│   ├── run_inference.py        # Inference entry point
│   ├── preprocess/             # Preprocessing scripts (valmark generation, excluded due to market edge)
│   ├── VA_markers_final.xlsx   # manually curated marker file (excluded from repo)
│   └── src/
│       ├── data_loader.py       # I/O and padding utilities
│       ├── dataset.py           # PyTorch Dataset wrapper
│       ├── dataset_balanced.py  # Balanced sampling for rare events
│       ├── model.py             # Architectures: TCN, Transformer, Longformer
│       └── train_utils.py       # Training loops, loss functions, evaluation
├── models/                    # Saved model checkpoints (excluded)
├── markers_example.xlsx       # Dummy template illustrating marker format
├── .gitignore
└── README.md                  # Project overview (this file)
```

## .gitignore

```gitignore
# Exclude proprietary marker data and detailed scripts
dev/VA_markers_final.xlsx
dev/preprocess/valmark/
# Exclude model checkpoints and large data directories
dev/models/
VAmodel/data/
```

## Prerequisites

* Python 3.8+
* PyArrow, NumPy, Pandas, PyTorch, Transformers, scikit-learn, tqdm, PyYAML

Install via:

```bash
pip install -r requirements.txt
```

## Installation

```bash
git clone git@github.com:<your-username>/VAmodel.git
cd VAmodel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All settings live in `dev/config.yaml`. Key sections:

```yaml
data:
  features:
    - Last
    - Volume
    - unique_id
  label_column: [label1, label2]
  allowed_values: [0, 2, 3, 4, 5, 6, 7, 8, -2, -3, -4, -5, -6, -7, -8]
  seq_length: 2048
  train_val_split: 0.8
  batch_size: 32
  training_data_dir_2: ../data/data_training
  inference_data_dir: ../data/data_training
  inference_results_dir: ../data/data_verdict
  increase: 0.2

model:
  type: tcn             # or transformer / longformer
  d_model: 128
  num_encoder_layers: 4  # for transformer/longformer
  nhead: 8               # for transformer/longformer
  dim_feedforward: 512
  dropout: 0.1
  tcn_kernel_size: 3
  tcn_padding: same
  tcn_num_levels: 4
  tcn_channel_growth: false
  attention_window: 2048  # for Longformer

training:
  max_lr: 1e-3
  epochs: 50
  weight_decay: 1e-2
  save_dir: ./models
  checkpoint_dir: ./models
  patience: 10
```

Adjust paths and hyperparameters as needed.

## Data Preparation

1. Place raw `.parquet` market files in `data/data_edit/`.
2. **Proprietary markers:** your own `VA_markers_final.xlsx` resides at `dev/VA_markers_final.xlsx` and detailed scripts live in `dev/preprocess/valmark/`, but both are excluded by `.gitignore`.
3. Use the included dummy template `markers_example.xlsx` to understand the expected format for marker inputs.
4. Run any custom preprocessing:

   ```bash
   python dev/preprocess/valmark/preprocess.py --markers dev/markers_example.xlsx --input ../data/data_edit --output ../data/data_training
   ```
5. Move processed files to `data/data_training/` for training and inference.

## Training

```bash
cd dev
python run_training.py --cfg config.yaml
```

Checkpoints will be written to the `dev/models/` directory (excluded).

## Inference

```bash
cd dev
python run_inference.py --cfg config.yaml
```

Results (`*_results.parquet`) are saved to `data/data_verdict/`.

## Development

* **Marker generation:** See `dev/preprocess/valmark/` for reverse stochastic polynomial scripts (excluded from repo, consult owner).
* **Model Architectures:** Defined in `dev/src/model.py` (TCN, Transformer, Longformer).
* **Data Loaders:** Utilities in `dev/src/data_loader.py`, `dataset.py`, and `dataset_balanced.py`.
* **Training/Evaluation:** Core logic in `dev/src/train_utils.py`.

Feel free to explore and extend to new features or architectures.

## Contributing

This is a personal research project. Pull requests are welcome for enhancements, bug fixes, or performance improvements.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
