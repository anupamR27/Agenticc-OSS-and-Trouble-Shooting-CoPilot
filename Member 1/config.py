"""Configuration for Member 1 anomaly detection workflow."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = Path("/Users/swinzal/Downloads/ds1_processed.csv")
PROCESSED_TRAIN_PATH = DATA_DIR / "train_processed.csv"
PROCESSED_TEST_PATH = DATA_DIR / "test_processed.csv"

ANOMALY_SCORES_PATH = OUTPUT_DIR / "anomaly_scores.csv"
FEATURE_REPORT_PATH = REPORT_DIR / "feature_report.csv"
MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
PREPROCESSING_PIPELINE_PATH = MODEL_DIR / "preprocessing_pipeline.pkl"

RANDOM_STATE = 42
SAMPLE_SIZE = 2000
TEST_SIZE = 0.20

TARGET_COLUMN = None
ID_COLUMNS = ["record_id", "cell_id"]
TIME_COLUMNS = ["timestamp"]
CATEGORICAL_COLUMNS = ["cell_type", "slice_type"]

KPI_KEYWORDS = [
    "throughput",
    "latency",
    "packet_loss",
    "handover",
    "rsrp",
    "rsrq",
    "prb",
    "active_users",
    "sla",
    "spectral_efficiency",
]

CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 0.0

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.05,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RISK_BINS = [0.0, 0.30, 0.60, 1.0]
RISK_LABELS = ["Healthy", "Moderate Risk", "High Risk"]
