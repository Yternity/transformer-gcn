# transformer-gcn

Minimal code package for the vegetation community classification and prediction experiments reported in the manuscript.

This repository is intentionally limited to the code needed for manuscript review. Large data files, trained weights, generated outputs, figures, manuscripts, and local working notes are excluded.

## Contents

- `train.py`: training entry point for the Transformer-GCN model and baseline neural models.
- `predict.py`: prediction entry point for chunked large-scale inference.
- `train_rf.py`: random-forest baseline entry point.
- `core/`: model definitions, graph construction, training, prediction, and evaluation utilities.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

`torch` and `torch-geometric` installation can depend on the local CUDA/Python version. If needed, install them from the official PyTorch and PyTorch Geometric instructions first, then install the remaining requirements.

## Quick Test

Run the included synthetic-data smoke test:

```bash
python quick_test.py
```

Expected output:

```text
quick_test passed: Transformer-GCN forward pass output shape = (6, 4)
```

This test imports the main Transformer-GCN model, builds a small synthetic graph, and runs one forward pass. It does not require the manuscript data files.

## Expected Local Data Layout

The data are not included in this repository. Place local data and model artifacts under:

```text
data/
  ours_data.pt
  prediction_set.csv
outputs/
  results/
```

Default paths are defined in `core/config.py`.

## Run

Train the main model:

```bash
python train.py --model-name ours --seed 42 --epochs 1000 --patience 50
```

Run prediction:

```bash
python predict.py --chunk-size 50000 --grid-size 0.4 --buffer-cells 1
```

Train the random-forest baseline:

```bash
python train_rf.py
```

Outputs are written to `outputs/`, which is ignored by Git.

## License

This code is distributed under the MIT License. See `LICENSE`.
