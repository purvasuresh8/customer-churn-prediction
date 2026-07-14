from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# DATA PATHS
# =====================================================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

RAW_DATA_FILE = RAW_DATA_DIR / "customer_churn.csv"

CLEANED_DATA_FILE = (
    PROCESSED_DATA_DIR / "cleaned_data.csv"
)

FEATURE_ENGINEERED_FILE = (
    PROCESSED_DATA_DIR /
    "feature_engineered_data.csv"
)

PREDICTIONS_FILE = (
    PREDICTIONS_DIR /
    "churn_predictions.csv"
)

RETENTION_PLAN_FILE = (
    PREDICTIONS_DIR /
    "retention_plan.csv"
)

# =====================================================
# MODEL PATHS
# =====================================================

MODELS_DIR = BASE_DIR / "models"

LOGISTIC_MODEL_PATH = (
    MODELS_DIR /
    "logistic_regression.pkl"
)

RANDOM_FOREST_MODEL_PATH = (
    MODELS_DIR /
    "random_forest.pkl"
)

# =====================================================
# REPORT PATHS
# =====================================================

REPORTS_DIR = BASE_DIR / "reports"

MODEL_COMPARISON_REPORT = (
    REPORTS_DIR /
    "model_comparison.csv"
)

PERFORMANCE_REPORT = (
    REPORTS_DIR /
    "model_performance_report.pdf"
)

BUSINESS_REPORT = (
    REPORTS_DIR /
    "churn_insights_report.pdf"
)

# =====================================================
# VISUALIZATION PATHS
# =====================================================

VISUALIZATION_DIR = BASE_DIR / "visualizations"

CONFUSION_MATRIX_DIR = (
    VISUALIZATION_DIR /
    "confusion_matrices"
)

ROC_CURVE_DIR = (
    VISUALIZATION_DIR /
    "roc_curves"
)

FEATURE_IMPORTANCE_DIR = (
    VISUALIZATION_DIR /
    "feature_importance"
)

SHAP_DIR = (
    VISUALIZATION_DIR /
    "shap"
)

# =====================================================
# LOGGING
# =====================================================

LOG_DIR = BASE_DIR / "logs"

LOG_FILE = LOG_DIR / "pipeline.log"

# =====================================================
# MACHINE LEARNING SETTINGS
# =====================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "Churn"

ID_COLUMN = "customerID"

# =====================================================
# MODEL PARAMETERS
# =====================================================

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": RANDOM_STATE
}

# =====================================================
# CHURN RISK THRESHOLDS
# =====================================================

HIGH_RISK_THRESHOLD = 0.80

MEDIUM_RISK_THRESHOLD = 0.50

LOW_RISK_THRESHOLD = 0.00

# =====================================================
# CREATE REQUIRED DIRECTORIES
# =====================================================

REQUIRED_DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    VISUALIZATION_DIR,
    LOG_DIR
]
