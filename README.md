# Bus-Probe Traffic Speed Estimation

This repository contains the implementation of a graph-based spatiotemporal model for estimating citywide traffic speed from public transit bus observations.

The method uses bus-derived speed time series as input and learns to estimate reference traffic speed profiles over a road-segment graph. The model combines historical feature construction, a stack of spatiotemporal encoder blocks (temporal convolution, temporal self-attention, spatial attention, and a diffusion graph convolution), and a gated dual-branch decoder that separates a coarse traffic baseline from localized residual corrections.

## Repository structure

```text
src/
  config.py
  main.py

  data/
    loaders.py

  preprocessing/
    alignment.py
    missing.py
    scaling.py
    prepare.py

  models/
    factory.py
    dual_branch_stgcnn.py

  training/
    trainer.py
    losses.py

  evaluation/
    metrics.py
    evaluator.py
    exporter.py

  utils/
    checkpoint.py
    device.py
    splitting.py
    windowing.py
```

## Data

The code expects three input files:

```text
data/
  stib_speeds.csv
  reference_speeds.csv
  adjacency.csv
```

`stib_speeds.csv` contains bus-derived segment-level speed observations.

`reference_speeds.csv` contains reference speed profiles used as training targets.

`adjacency.csv` contains the road-segment adjacency matrix. Rows and columns should correspond to segment identifiers.

The first column of the speed files must contain timestamps. The remaining columns must correspond to segment identifiers.

Commercial API outputs and private datasets are not included in this repository. Users should provide their own reference speed data or use a synthetic/sample file with the same schema.


## Outputs are written to:

```text
outputs/
  predictions.csv
  model.pt
  stib_test.csv
  reference_test.csv
  prediction_test.csv
  metrics_per_segment.csv
```

## Model

The implemented model is `DualBranchSTGCNN`, defined in:

```text
src/models/dual_branch_stgcnn.py
```

Input shape:

```text
[batch_size, input_window, num_segments]
```

Output shape:

```text
[batch_size, num_segments]
```

