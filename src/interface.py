import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from catboost import Pool

# ============================================================
# CONFIG
# ============================================================
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "catboost_model.pkl")
DEFAULT_TRAIN_FILE = os.getenv("TRAIN_FILE", "materials_preorder_train_80.xlsx")

APP_TITLE = "Long-Lead Item Predictor (CatBoost)"
APP_SUBTITLE = "Predict lead-time (regression) and classify Long-Lead using a threshold."

# A reasonable default set of "core" fields to show first (edit as you like)
DEFAULT_CORE_FIELDS = [
    "category",
    "supplier_region",
    "shipping_mode",
    "incoterm",
    "import_flag",
    "unit_cost_usd",
    "min_order_qty",
    "annual_demand_units",
    "distance_km",
    "geopolitical_risk_index_0_100",
    "capacity_utilization_pct",
    "backlog_index_0_100",
    "historical_lead_time_mean_days",
    "historical_lead_time_std_days",
]

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource
def load_model(model_path: str):
    """Load CatBoost model once per session."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Expected your saved joblib model (e.g., catboost_model.pkl)."
        )
    return joblib.load(model_path)


@st.cache_data
def load_train_features(train_file: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load the training file and return ONLY feature columns (assumes last col = target).
    If nrows is None -> reads the entire file (better defaults, slower).
    """
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            "Provide the file or set TRAIN_FILE env var."
        )

    df = pd.read_excel(train_file, engine="openpyxl", nrows=nrows)
    if df.shape[1] < 2:
        raise ValueError("Training file must have at least 2 columns (features + target).")

    X = df.iloc[:, :-1].copy()  # last column = target (as in your train.py assumption)
    return X


# ============================================================
# DEFAULTS + SCHEMA INFERENCE
# ============================================================
def compute_defaults_from_data(X: pd.DataFrame) -> dict:
    """
    Compute stable defaults:
      - numeric: median
      - categorical/text: mode (most frequent)
    """
    defaults = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            defaults[c] = float(s.median()) if s.notna().any() else 0.0
        else:
            m = s.dropna().astype(str).mode()
            defaults[c] = m.iloc[0] if len(m) else ""
    return defaults


def infer_feature_config(X_template: pd.DataFrame, max_categories: int = 30) -> dict:
    """
    Infer which features are numeric vs categorical and gather dropdown categories.
    """
    cat_cols = X_template.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_template.columns if c not in cat_cols]

    categories = {}
    for c in cat_cols:
        vals = (
            X_template[c]
            .astype(str)
            .replace("nan", np.nan)
            .dropna()
            .value_counts()
            .head(max_categories)
            .index
            .tolist()
        )
        categories[c] = vals

    return {
        "numeric": num_cols,
        "categorical": cat_cols,
        "categories": categories,
    }


def apply_defaults(user_values: dict, defaults: dict, feature_order: list[str]) -> dict:
    """
    Merge user input into defaults.
    Any None or blank string becomes default.
    """
    final = {}
    for feat in feature_order:
        v = user_values.get(feat, None)
        if v is None:
            final[feat] = defaults.get(feat, "")
        elif isinstance(v, str) and v.strip() == "":
            final[feat] = defaults.get(feat, "")
        else:
            final[feat] = v
    return final


# ============================================================
# UI BUILDING (OPTIONAL INPUTS)
# ============================================================
def feature_input_widget(
    feat: str,
    feature_cfg: dict,
    defaults: dict,
    key_prefix: str = ""
):
    """
    Render a widget for one feature.
    Returns either:
      - a user-provided value
      - or None meaning "use default"
    """
    default_val = defaults.get(feat, "")

    # Numeric feature: allow "Use default" checkbox
    if feat in feature_cfg["numeric"]:
        use_def = st.checkbox(
            f"Use default for {feat}",
            value=True,
            key=f"{key_prefix}{feat}_use_default"
