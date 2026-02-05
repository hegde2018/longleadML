import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from catboost import Pool

# -----------------------------
# CONFIG (match your train.py)
# -----------------------------
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "catboost_model.pkl")

# Used only to infer feature names + data types for UI.
# For public deployment, consider replacing this with a schema JSON instead.
DEFAULT_TRAIN_FILE = os.getenv("TRAIN_FILE", "materials_preorder_train_80.xlsx")

APP_TITLE = "Long-Lead Item Predictor (CatBoost)"
APP_SUBTITLE = (
    "Predict lead-time (regression) and classify whether an order is Long-Lead using a threshold."
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Expected your saved joblib model (e.g., catboost_model.pkl)."
        )
    model = joblib.load(model_path)
    return model


@st.cache_data
def load_train_template(train_file: str, nrows: int = 200) -> pd.DataFrame:
    """
    Load a small sample of training data to infer:
      - feature column names (all columns except last assumed target)
      - dtypes (numeric vs categorical)
      - example values & common categories (for dropdowns)
    """
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            "This UI uses the training file to infer input fields. "
            "Either provide the file or set TRAIN_FILE env var. "
            "For public deployment, switch to a schema JSON (see notes)."
        )

    df = pd.read_excel(train_file, engine="openpyxl", nrows=nrows)
    if df.shape[1] < 2:
        raise ValueError("Training file must contain at least 2 columns (features + target).")

    X = df.iloc[:, :-1].copy()  # last column = target (as in your train.py)
    return X


def infer_feature_config(X_template: pd.DataFrame, max_categories: int = 30):
    """
    Create a config dict like:
    {
      "numeric": [col1, col2],
      "categorical": [col3, col4],
      "categories": {col3: [...], col4: [...]},
      "defaults": {col: value}
    }
    """
    cat_cols = X_template.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_template.columns if c not in cat_cols]

    categories = {}
    defaults = {}

    for c in X_template.columns:
        # default value: first non-null
        non_null = X_template[c].dropna()
        defaults[c] = non_null.iloc[0] if len(non_null) else ("" if c in cat_cols else 0.0)

        if c in cat_cols:
            # top categories from sample
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
        "defaults": defaults,
    }


def build_form(feature_cfg):
    """
    Build Streamlit inputs for each feature.
    Returns dict {feature_name: value}
    """
    st.subheader("Enter Order Details")

    values = {}
    cols = st.columns(2)

    all_features = feature_cfg["numeric"] + feature_cfg["categorical"]

    for i, feat in enumerate(all_features):
        target_col = cols[i % 2]
        default_val = feature_cfg["defaults"].get(feat, "")

        with target_col:
            if feat in feature_cfg["numeric"]:
                # Use float input (works for int too)
                try:
                    dv = float(default_val) if default_val != "" else 0.0
                except Exception:
                    dv = 0.0
                values[feat] = st.number_input(feat, value=dv, format="%.4f")
            else:
                options = feature_cfg["categories"].get(feat, [])
                # If we have category options from sample data, show dropdown; else text input
                if options:
                    # best-effort default index
                    default_str = str(default_val) if default_val is not None else ""
                    idx = options.index(default_str) if default_str in options else 0
                    values[feat] = st.selectbox(feat, options=options, index=idx)
                else:
                    values[feat] = st.text_input(feat, value=str(default_val) if default_val is not None else "")

    return values


def predict(model, input_dict, cat_cols):
    """
    CatBoost predicts from pandas DataFrame.
    Ensure categorical cols are string type.
    """
    X = pd.DataFrame([input_dict])

    # Ensure cat cols are string (CatBoost handles them as categorical)
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)

    # Predict numeric output
    y_pred = float(model.predict(X)[0])
    return y_pred


def global_feature_importance(model, feature_names):
    """
    Get CatBoost global feature importance.
    """
    try:
        importances = model.get_feature_importance()
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        df = df.sort_values("importance", ascending=False)
        return df
    except Exception:
        return None


def explain_single_prediction_shap(model, X_row: pd.DataFrame, cat_cols):
    """
    Per-row SHAP values using CatBoost.
    Returns DataFrame with feature -> shap_value.
    """
    # identify cat feature indices from X_row
    cat_feature_indices = [X_row.columns.get_loc(c) for c in cat_cols if c in X_row.columns]

    pool = Pool(X_row, cat_features=cat_feature_indices)

    # ShapValues returns array: (n_samples, n_features + 1) where last is expected value
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = np.array(shap_vals)

    # single row
    row_vals = shap_vals[0]
    feature_vals = row_vals[:-1]
    expected_value = row_vals[-1]

    df = pd.DataFrame({
        "feature": X_row.columns,
        "shap_value": feature_vals
    }).sort_values("shap_value", key=lambda s: np.abs(s), ascending=False)

    return df, float(expected_value)


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
    st.title("📦 " + APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.sidebar:
        st.header("Settings")

        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
        train_file = st.text_input("Train file (for UI schema)", value=DEFAULT_TRAIN_FILE)

        st.divider()
        st.subheader("Long-Lead Threshold")
        threshold = st.number_input(
            "Threshold (same units as model output, e.g., days)",
            value=float(os.getenv("LONG_LEAD_THRESHOLD", "30")),
            step=1.0
        )

        show_global_importance = st.toggle("Show global feature importance", value=True)
        show_shap = st.toggle("Explain this prediction (SHAP)", value=False)

        st.divider()
        st.markdown("### Notes")
        st.markdown(
            "- Your model is a **regressor**. We classify **Long-Lead** using the threshold.\n"
            "- For public deployment, consider using a schema JSON instead of loading the training file."
        )

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Load template to infer features
    try:
        X_template = load_train_template(train_file, nrows=200)
        feature_cfg = infer_feature_config(X_template)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Build UI form
    input_dict = build_form(feature_cfg)

    st.divider()
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict_btn:
        try:
            y_pred = predict(model, input_dict, feature_cfg["categorical"])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        is_long_lead = y_pred >= threshold

        # Results
        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted value", f"{y_pred:.3f}")
        with c2:
            st.metric("Threshold", f"{threshold:.3f}")
        with c3:
            st.metric("Long-Lead?", "✅ YES" if is_long_lead else "❌ NO")

        # Simple indicator bar
        ratio = min(max(y_pred / threshold, 0.0), 2.0) if threshold != 0 else 0.0
        st.progress(min(ratio / 2.0, 1.0))

        with st.expander("Show input payload"):
            st.json(input_dict)

        # Global feature importance
        if show_global_importance:
            fi = global_feature_importance(model, list(X_template.columns))
            if fi is not None:
                st.subheader("Global Feature Importance")
                st.bar_chart(fi.head(15).set_index("feature")["importance"])
                with st.expander("See full importance table"):
                    st.dataframe(fi, use_container_width=True)
            else:
                st.info("Could not compute global feature importance for this model.")

        # SHAP explanation for this prediction
        if show_shap:
            try:
                X_row = pd.DataFrame([input_dict])
                for c in feature_cfg["categorical"]:
                    if c in X_row.columns:
                        X_row[c] = X_row[c].astype(str)

                shap_df, expected = explain_single_prediction_shap(model, X_row, feature_cfg["categorical"])
                st.subheader("Why this prediction? (SHAP)")
                st.caption(f"Expected value (baseline): {expected:.3f}")

                st.bar_chart(shap_df.head(15).set_index("feature")["shap_value"])
                with st.expander("See SHAP table"):
                    st.dataframe(shap_df, use_container_width=True)
            except Exception as e:
                st.info(f"Could not compute SHAP for this prediction: {e}")


if __name__ == "__main__":
    main()