from pathlib import Path

# --------------------------------------------------
# Roots
# --------------------------------------------------

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent

# --------------------------------------------------
# Data paths
# --------------------------------------------------

STIB_CSV = PROJECT_ROOT / "STIB_timeseries_collector" / "Merged_Dataset" / "0.STIB_speeds.csv"
GOOGLE_CSV = PROJECT_ROOT / "Google_timeseries_collector" / "Merged_Dataset" / "0.speeds.csv"

# --------------------------------------------------
# Results
# --------------------------------------------------

RESULTS_ROOT = PROJECT_ROOT / "Model_SIG2026" / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

FINAL_PREDICTION_CSV = RESULTS_ROOT / "google_speed_predictions.csv"

# --------------------------------------------------
# Training config
# --------------------------------------------------

WINDOW_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

USE_MPS_IF_AVAILABLE = True

# --------------------------------------------------
# Split config
# --------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15   # test = 1 - train - val
MIN_GOOGLE_OBS_PER_ROW = 1  # Minimum required observed Google values per timestamp.

# Split strategy
SPLIT_STRATEGY = "by_days"   # options: "ratio", "by_days"

# If SPLIT_STRATEGY == "by_days":
TEST_DAYS = 3               # last N full days as test
VAL_DAYS = 2                # days right before test as validation

# --------------------------------------------------
# Model checkpoints
# --------------------------------------------------
CHECKPOINT_ROOT = PROJECT_ROOT / "Model_SIG2026"
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_CHECKPOINT = CHECKPOINT_ROOT / "cnn_trained model.pt"

# --------------------------------------------------
# Temporal feature configuration
# --------------------------------------------------
RECENT_STEPS = 4              # intra-day frames (same day only)

USE_DAILY_LAG = True          # t - 1 day
USE_WEEKLY_LAG = True         # t - 7 days

USE_SIMILAR_DAY = True        # similarity-based historical days
NUM_SIMILAR_DAYS = 2          # how many similar weekdays to include

PADDING_VALUE = 0.0

# --------------------------------------------------
# Early stopping / checkpoint selection
# --------------------------------------------------
EARLY_STOPPING = True
PATIENCE = 5
MIN_DELTA = 0.0  # minimum change in validation loss to qualify as an improvement

# --------------------------------------------------
# Normalization
# --------------------------------------------------
USE_NORMALIZATION = True
NORMALIZATION_EPS = 1e-6
NORMALIZE_TARGET = True  # Google speeds
NORMALIZE_INPUT = True   # STIB speeds
LAMBDA_SMOOTH = 0.0

VAR_SCALE_RAW = 5.0
VAR_SCALE_NORM = 2.0

# --------------------------------------------------
# Loss
# --------------------------------------------------
LAMBDA_SMOOTH = 0.0   # keep OFF if you do not want smoothing
LAMBDA_DELTA = 0.5    # start with 0.3~1.0 and tune

# Loss selection (maintainable; main.py must not manage per-loss params)
LOSS_NAME = "masked_mse"

LOSS_PARAMS = {
    "masked_mse": {
        "reduction": "mean",
    }
}

# --------------------------------------------------
# Test exports
# --------------------------------------------------
TEST_EXPORT_DIRNAME = "test_exports"
TEST_EXPORT_ROOT = RESULTS_ROOT / TEST_EXPORT_DIRNAME
TEST_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

# Choose model: "cnn", "tf_fusion", "spike_mixture", "film_cnn", "refiner_cnn", "lstm", "graph_wavenet_no_adj", "arima"
MODEL_NAME = "refiner_cnn"

# Model-specific kwargs. Only the selected model will use its own kwargs.
MODEL_KWARGS = {
    "cnn": {
        # keep empty or add cnn-specific options later
    },
    "tf_fusion": {
        "use_log_mag": False,
        "use_attention_pool": False,
        "pool_type": "max",
        "hidden": 64,
    },
    "spike_mixture": {
        "hidden": 64,
        "delta_scale": 2.0,   # if target normalized; try 2.0 -> 3.0
        "pool_type": "max",
        "topk": 4,
        "use_topk_pool": True,
    },
    "film_cnn": {
        "hidden": 64,
        "cond_dim": 8,
        "topk": 4,
    },
    "refiner_cnn": {
            "hidden": 64,
            "topk": 4,
            # delta_scale is optional; if omitted, factory uses var_scale computed in main
            # "delta_scale": 2.0,
        },
    "arima": {
        "order": (8, 0, 1),
        "min_obs": 80,
        "fallback": "last",
        "verbose": True,
    },
}
