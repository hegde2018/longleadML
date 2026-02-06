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
        )
        if use_def:
            st.caption(f"Default: {default_val}")
            return None

        # User enters explicit value
        try:
            dv = float(default_val) if default_val != "" else 0.0
        except Exception:
            dv = 0.0
        return st.number_input(feat, value=dv, format="%.4f", key=f"{key_prefix}{feat}_num")

    # Categorical feature: dropdown with "__DEFAULT__" option
    options = feature_cfg["categories"].get(feat, [])
    if options:
        choice = st.selectbox(
            feat,
            options=["__DEFAULT__"] + options,
            index=0,
            key=f"{key_prefix}{feat}_cat"
        )
        if choice == "__DEFAULT__":
            st.caption(f"Default: {default_val}")
            return None
        return choice

    # Fallback text input (optional)
    txt = st.text_input(
        feat,
        value="",
        placeholder=f"Default: {default_val}",
        key=f"{key_prefix}{feat}_txt"
    )
    return txt if txt.strip() else None


def build_form(feature_order: list[str], feature_cfg: dict, defaults: dict, core_fields: list[str]):
    """
    Build the UI:
      - Core fields displayed first
      - Advanced fields under an expander
    Returns dict {feature: value or None}
    """
    st.subheader("Enter Order Details (optional fields will use defaults)")
    values = {}

    # --- Core fields ---
    st.markdown("### Core fields")
    core = [f for f in core_fields if f in feature_order]
    if not core:
        st.info("No core fields selected. Choose core fields in the sidebar.")
    else:
        cols = st.columns(2)
        for i, feat in enumerate(core):
            with cols[i % 2]:
                values[feat] = feature_input_widget(feat, feature_cfg, defaults, key_prefix="core_")

    # --- Advanced fields ---
    advanced = [f for f in feature_order if f not in core]
    with st.expander(f"Advanced fields ({len(advanced)} hidden by default)"):
        cols = st.columns(2)
        for i, feat in enumerate(advanced):
            with cols[i % 2]:
                values[feat] = feature_input_widget(feat, feature_cfg, defaults, key_prefix="adv_")

    return values


# ============================================================
# MODEL UTILS
# ============================================================
def predict(model, user_values: dict, feature_order: list[str], cat_cols: list[str], defaults: dict) -> float:
    """
    Apply defaults -> build DataFrame in correct order -> predict.
    """
    payload = apply_defaults(user_values, defaults, feature_order)
    X = pd.DataFrame([payload])[feature_order]  # enforce exact column order

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)

    y_pred = float(model.predict(X)[0])
    return y_pred


def global_feature_importance(model, feature_names: list[str]):
    """Global feature importance from CatBoost (if available)."""
    try:
        importances = model.get_feature_importance()
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False)
    except Exception:
        return None


def explain_single_prediction_shap(model, X_row: pd.DataFrame, cat_cols: list[str]):
    """Per-row SHAP values using CatBoost Pool."""
    cat_feature_indices = [X_row.columns.get_loc(c) for c in cat_cols if c in X_row.columns]
    pool = Pool(X_row, cat_features=cat_feature_indices)

    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = np.array(shap_vals)

    row_vals = shap_vals[0]
    feature_vals = row_vals[:-1]
    expected_value = row_vals[-1]

    df = pd.DataFrame({"feature": X_row.columns, "shap_value": feature_vals})
    df = df.sort_values("shap_value", key=lambda s: np.abs(s), ascending=False)
    return df, float(expected_value)


# ============================================================
# STREAMLIT APP
# ============================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
    st.title("📦 " + APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.sidebar:
        st.header("Settings")

        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
        train_file = st.text_input("Train file (schema + defaults)", value=DEFAULT_TRAIN_FILE)

        st.divider()
        st.subheader("Defaults quality vs Speed")
        use_full_file = st.toggle(
            "Use full Excel file to compute defaults (more accurate, slower)",
            value=True
        )
        nrows = None if use_full_file else 500  # faster defaults if you want

        st.divider()
        st.subheader("Long-Lead Threshold")
        threshold = st.number_input(
            "Threshold (same units as model output, e.g., days)",
            value=float(os.getenv("LONG_LEAD_THRESHOLD", "30")),
            step=1.0
        )

        st.divider()
        st.subheader("UI: Core fields")
        # We need feature list to validate core fields, but we can still accept a selection here later
        core_fields_text = st.text_area(
            "Core fields (one per line). Leave blank to use defaults list.",
            value="",
            height=150
        )
        show_global_importance = st.toggle("Show global feature importance", value=True)
        show_shap = st.toggle("Explain this prediction (SHAP)", value=False)

        st.divider()
        st.markdown("### Notes")
        st.markdown(
            "- Any field left as **default** is filled automatically.\n"
            "- The app sends the model the full feature row in the correct column order."
        )

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Load training features (for schema + defaults)
    try:
        X_all = load_train_features(train_file, nrows=nrows)
    except Exception as e:
        st.error(str(e))
        st.stop()

    feature_order = list(X_all.columns)

    # Infer schema + compute defaults
    feature_cfg = infer_feature_config(X_all)
    defaults = compute_defaults_from_data(X_all)

    # Determine core fields list
    if core_fields_text.strip():
        core_fields = [x.strip() for x in core_fields_text.splitlines() if x.strip()]
    else:
        core_fields = DEFAULT_CORE_FIELDS

    # Build UI
    user_values = build_form(feature_order, feature_cfg, defaults, core_fields)

    st.divider()
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict_btn:
        try:
            y_pred = predict(
                model=model,
                user_values=user_values,
                feature_order=feature_order,
                cat_cols=feature_cfg["categorical"],
                defaults=defaults
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        is_long_lead = y_pred >= threshold

        st.subheader("Prediction Result")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted value", f"{y_pred:.3f}")
        with c2:
            st.metric("Threshold", f"{threshold:.3f}")
        with c3:
            st.metric("Long-Lead?", "✅ YES" if is_long_lead else "❌ NO")

        ratio = min(max(y_pred / threshold, 0.0), 2.0) if threshold != 0 else 0.0
        st.progress(min(ratio / 2.0, 1.0))

        # Show final model payload (after defaults)
        final_payload = apply_defaults(user_values, defaults, feature_order)
        with st.expander("Show final payload sent to model (after defaults)"):
            st.json(final_payload)

        # Global feature importance
        if show_global_importance:
            fi = global_feature_importance(model, feature_order)
            if fi is not None:
                st.subheader("Global Feature Importance")
                st.bar_chart(fi.head(15).set_index("feature")["importance"])
                with st.expander("See full importance table"):
                    st.dataframe(fi, use_container_width=True)
            else:
                st.info("Could not compute global feature importance for this model.")

        # SHAP explanation
        if show_shap:
            try:
                X_row = pd.DataFrame([final_payload])[feature_order]
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