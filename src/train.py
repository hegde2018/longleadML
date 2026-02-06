import json
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# TRAINING SCRIPT (Classification)
# ------------------------------------------------------------
# Goal:
#   - Train a CatBoostClassifier to predict is_long_lead_item (0/1)
#   - NEVER use actual_lead_time_days_DO_NOT_USE_FOR_TRAINING as a feature
#   - Save model + schema.json (feature order, cat indices, defaults, categories)
# ============================================================

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_FILE = "materials_preorder_train_80.xlsx"
VALID_FILE = "materials_preorder_test_20.xlsx"

TARGET_COL = "is_long_lead_item"  # binary label column
TEST_ONLY_COLS = ["actual_lead_time_days_DO_NOT_USE_FOR_TRAINING"]

# Optional: drop ID-like columns to reduce overfitting / improve generalization
# (IDs/names often have very high cardinality)
DROP_COLS = [
    # "material_id",
    # "material_name",
]

MODEL_OUT_PKL = "catboost_model.pkl"
MODEL_OUT_CBM = "catboost_model.cbm"
SCHEMA_OUT_JSON = "schema.json"

RANDOM_SEED = 42
MAX_CATEGORIES_PER_CAT_COL = 30


# -----------------------------
# Helpers
# -----------------------------

def compute_defaults(X: pd.DataFrame) -> dict:
    """Compute stable defaults for optional inference inputs.

    - Numeric: median (robust)
    - Categorical/text: mode (most frequent)
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


def compute_categories(X: pd.DataFrame, cat_cols: list[str], max_k: int = 30) -> dict:
    """Top-K frequent values per categorical feature (for UI dropdowns)."""
    categories = {}
    for c in cat_cols:
        vals = (
            X[c]
            .astype(str)
            .replace("nan", np.nan)
            .dropna()
            .value_counts()
            .head(max_k)
            .index
            .tolist()
        )
        categories[c] = vals
    return categories


def build_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Build X and y from a raw dataframe.

    Returns:
        X: features
        y: target label
        y_actual_days: optional actual lead time days (for evaluation only)
    """
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[TARGET_COL].astype(int)
    y_actual_days = df[TEST_ONLY_COLS[0]] if TEST_ONLY_COLS[0] in df.columns else None

    drop = [TARGET_COL] + [c for c in TEST_ONLY_COLS if c in df.columns] + [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop, errors="ignore").copy()

    return X, y, y_actual_days


def main():
    # -----------------------------
    # 1) Load data
    # -----------------------------
    train_df = pd.read_excel(TRAIN_FILE, engine="openpyxl")
    val_df = pd.read_excel(VALID_FILE, engine="openpyxl")

    # -----------------------------
    # 2) Build X/y explicitly by name (avoid 'last column' surprises)
    # -----------------------------
    X_train, y_train, _ = build_X_y(train_df)
    X_val, y_val, y_val_days = build_X_y(val_df)

    # -----------------------------
    # 3) Ensure features match (name + order)
    # -----------------------------
    if set(X_train.columns) != set(X_val.columns):
        missing_in_val = set(X_train.columns) - set(X_val.columns)
        extra_in_val = set(X_val.columns) - set(X_train.columns)
        raise ValueError(
            "Train/Validation feature columns do not match!\n"
            f"Missing in validation: {missing_in_val}\n"
            f"Extra in validation: {extra_in_val}\n"
            "Fix by using the same columns in both files (or adjust DROP_COLS/TEST_ONLY_COLS)."
        )

    # Use training column order as the canonical order
    X_val = X_val[X_train.columns]

    feature_names = list(X_train.columns)

    # -----------------------------
    # 4) Detect categorical columns (dtype-based)
    # -----------------------------
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    # Ensure categorical columns are strings
    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)

    print(f"Detected {len(cat_cols)} categorical columns")

    # -----------------------------
    # 5) Train classifier
    # -----------------------------
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        verbose=200,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=150,
    )

    # -----------------------------
    # 6) Evaluate
    # -----------------------------
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_val, proba)),
        "accuracy": float(accuracy_score(y_val, pred)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_val, pred).tolist(),
    }

    # Optional evaluation using actual lead-time days (test-only)
    if y_val_days is not None:
        # Ground truth long-lead label derived from actual days with a 30-day cutoff
        # (only for reporting; this does not affect training)
        y_true_from_days = (y_val_days >= 30).astype(int)
        metrics["agreement_with_actual_days_cutoff30"] = float(accuracy_score(y_true_from_days, pred))

    print("\n✅ Validation metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # -----------------------------
    # 7) Save model
    # -----------------------------
    joblib.dump(model, MODEL_OUT_PKL)
    model.save_model(MODEL_OUT_CBM)
    print(f"\n✅ Model saved to: {MODEL_OUT_PKL} and {MODEL_OUT_CBM}")

    # -----------------------------
    # 8) Save schema.json
    # -----------------------------
    defaults = compute_defaults(X_train)
    categories = compute_categories(X_train, cat_cols, max_k=MAX_CATEGORIES_PER_CAT_COL)

    schema = {
        "task": "classification",
        "target_col": TARGET_COL,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "cat_feature_indices": cat_feature_indices,
        "defaults": defaults,
        "categories": categories,
        "drop_cols": DROP_COLS,
        "test_only_cols": TEST_ONLY_COLS,
        "metrics": metrics,
        "notes": "Schema generated at training time to keep inference consistent.",
    }

    with open(SCHEMA_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"✅ Schema saved to: {SCHEMA_OUT_JSON}")


if __name__ == "__main__":
    main()