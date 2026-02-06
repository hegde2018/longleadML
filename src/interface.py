import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from catboost import Pool

# ============================================================
# STREAMLIT APP (Schema-driven)
# ------------------------------------------------------------
# - Loads model + schema.json
# - Builds a compact UI: Core fields + Advanced fields
# - Any skipped field uses defaults from schema
# - Uses schema's feature order + cat feature list to avoid mismatches
# ============================================================

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "catboost_model.pkl")
DEFAULT_SCHEMA_PATH = os.getenv("SCHEMA_PATH", "schema.json")

APP_TITLE = "Long-Lead Item Predictor (CatBoost)"
APP_SUBTITLE = "Predict probability of Long-Lead and classify using a probability cutoff."

# Edit this list to what you want users to see first.
DEFAULT_CORE_FIELDS = [
    "category",
    "supplier_region",
    "shipping_mode",
    "incoterm",
    "import_flag",
    "unit_cost_usd",
    "annual_demand_units",
    "distance_km",
]


@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_data
def load_schema(schema_path: str) -> dict:
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}. Run train.py to generate schema.json"
        )
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_defaults(user_values: dict, defaults: dict, feature_order: list[str]) -> dict:
    """Merge user inputs into defaults (None/blank -> default)."""
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


def infer_numeric_features(feature_order: list[str], cat_cols: list[str]) -> set:
    """Treat everything not in cat_cols as numeric for UI purposes."""
    cat_set = set(cat_cols)
    return set([f for f in feature_order if f not in cat_set])


def feature_input_widget(
    feat: str,
    defaults: dict,
    categories: dict,
    numeric_features: set,
    key_prefix: str = "",
):
    """Optional input widget; returns user value or None to use default."""
    default_val = defaults.get(feat, "")

    if feat in numeric_features:
        use_def = st.checkbox(
            f"Use default for {feat}",
            value=True,
            key=f"{key_prefix}{feat}_use_default",
        )
        if use_def:
            st.caption(f"Default: {default_val}")
            return None

        # user provides numeric value
        try:
            dv = float(default_val) if default_val != "" else 0.0
        except Exception:
            dv = 0.0
        return st.number_input(
            feat, value=dv, format="%.4f", key=f"{key_prefix}{feat}_num"
        )

    # categorical -> dropdown
    opts = categories.get(feat, [])
    if opts:
        choice = st.selectbox(
            feat,
            options=["__DEFAULT__"] + opts,
            index=0,
            key=f"{key_prefix}{feat}_cat",
        )
        if choice == "__DEFAULT__":
            st.caption(f"Default: {default_val}")
            return None
        return choice

    # fallback optional text
    txt = st.text_input(
        feat,
        value="",
        placeholder=f"Default: {default_val}",
        key=f"{key_prefix}{feat}_txt",
    )
    return txt if txt.strip() else None


def build_form(feature_order, defaults, categories, numeric_features, core_fields):
    st.subheader("Enter Order Details (optional fields will use defaults)")
    values = {}

    core = [c for c in core_fields if c in feature_order]
    advanced = [f for f in feature_order if f not in core]

    st.markdown("### Core fields")
    cols = st.columns(2)
    for i, feat in enumerate(core):
        with cols[i % 2]:
            values[feat] = feature_input_widget(
                feat, defaults, categories, numeric_features, key_prefix="core_"
            )

    with st.expander(f"Advanced fields ({len(advanced)} hidden by default)"):
        cols = st.columns(2)
        for i, feat in enumerate(advanced):
            with cols[i % 2]:
                values[feat] = feature_input_widget(
                    feat, defaults, categories, numeric_features, key_prefix="adv_"
                )

    return values


def predict_proba(model, payload: dict, feature_order: list[str], cat_cols: list[str]) -> float:
    """Predict probability of long-lead (class=1)."""
    X = pd.DataFrame([payload])[feature_order]

    # Ensure categorical columns are strings
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)

    proba = float(model.predict_proba(X)[:, 1][0])
    return proba


def global_feature_importance(model, feature_names):
    try:
        importances = model.get_feature_importance()
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False)
    except Exception:
        return None


def explain_single_prediction_shap(model, X_row: pd.DataFrame, cat_cols: list[str]):
    cat_feature_indices = [X_row.columns.get_loc(c) for c in cat_cols if c in X_row.columns]
    pool = Pool(X_row, cat_features=cat_feature_indices)

    shap_vals = np.array(model.get_feature_importance(pool, type="ShapValues"))
    row_vals = shap_vals[0]
    feature_vals = row_vals[:-1]
    expected_value = row_vals[-1]

    df = pd.DataFrame({"feature": X_row.columns, "shap_value": feature_vals})
    df = df.sort_values("shap_value", key=lambda s: np.abs(s), ascending=False)
    return df, float(expected_value)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
    st.title("📦 " + APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
        schema_path = st.text_input("Schema path", value=DEFAULT_SCHEMA_PATH)

        st.divider()
        st.subheader("Long-Lead Probability Cutoff")
        cutoff = st.number_input(
            "Cutoff (0–1)",
            value=float(os.getenv("LONG_LEAD_PROBA_CUTOFF", "0.5")),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )

        st.divider()
        st.subheader("UI: Core fields")
        core_fields_text = st.text_area(
            "Core fields (one per line). Leave blank to use defaults list.",
            value="",
            height=150,
        )

        show_global_importance = st.toggle("Show global feature importance", value=True)
        show_shap = st.toggle("Explain this prediction (SHAP)", value=False)

    # Load model + schema
    try:
        model = load_model(model_path)
        schema = load_schema(schema_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Schema-driven feature order + categoricals
    feature_order = schema["feature_names"]
    cat_cols = schema.get("cat_cols", [])
    defaults = schema.get("defaults", {})
    categories = schema.get("categories", {})

    numeric_features = infer_numeric_features(feature_order, cat_cols)

    # Core fields
    if core_fields_text.strip():
        core_fields = [x.strip() for x in core_fields_text.splitlines() if x.strip()]
    else:
        core_fields = DEFAULT_CORE_FIELDS

    with st.expander("Debug: Schema summary"):
        st.write("Task:", schema.get("task"))
        st.write("Target:", schema.get("target_col"))
        st.write("Total features:", len(feature_order))
        st.write("Categorical features:", len(cat_cols))
        st.write("First 10 features:", feature_order[:10])

    user_values = build_form(feature_order, defaults, categories, numeric_features, core_fields)

    st.divider()
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict_btn:
        final_payload = apply_defaults(user_values, defaults, feature_order)

        try:
            p_long = predict_proba(model, final_payload, feature_order, cat_cols)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        is_long = p_long >= cutoff

        st.subheader("Prediction Result")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("P(Long-Lead)", f"{p_long:.3f}")
        with c2:
            st.metric("Cutoff", f"{cutoff:.2f}")
        with c3:
            st.metric("Long-Lead?", "✅ YES" if is_long else "❌ NO")

        # Progress bar maps probability to 0..1
        st.progress(min(max(p_long, 0.0), 1.0))

        with st.expander("Show final payload sent to model"):
            st.json(final_payload)

        if show_global_importance:
            fi = global_feature_importance(model, feature_order)
            if fi is not None:
                st.subheader("Global Feature Importance")
                st.bar_chart(fi.head(15).set_index("feature")["importance"])
                with st.expander("Full importance table"):
                    st.dataframe(fi, use_container_width=True)

        if show_shap:
            try:
                X_row = pd.DataFrame([final_payload])[feature_order]
                for c in cat_cols:
                    if c in X_row.columns:
                        X_row[c] = X_row[c].astype(str)

                shap_df, expected = explain_single_prediction_shap(model, X_row, cat_cols)
                st.subheader("Why this prediction? (SHAP)")
                st.caption(f"Expected value (baseline): {expected:.3f}")
                st.bar_chart(shap_df.head(15).set_index("feature")["shap_value"])
                with st.expander("SHAP table"):
                    st.dataframe(shap_df, use_container_width=True)
            except Exception as e:
                st.info(f"Could not compute SHAP: {e}")


if __name__ == "__main__":
    main()