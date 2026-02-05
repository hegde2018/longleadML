import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# CONFIG: Update these filenames
# -----------------------------
TRAIN_FILE = "materials_preorder_train_80.xlsx"
VALID_FILE = "materials_preorder_test_20.xlsx"

MODEL_OUT_PKL = "catboost_model.pkl"
MODEL_OUT_CBM = "catboost_model.cbm"

RANDOM_SEED = 42

# -----------------------------
# 1) Load train + validation data
# -----------------------------
train_data = pd.read_excel(TRAIN_FILE, engine="openpyxl")
val_data   = pd.read_excel(VALID_FILE, engine="openpyxl")

# -----------------------------
# 2) Split into X/y
# Assumption: last column is target in BOTH files
# -----------------------------
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_val = val_data.iloc[:, :-1]
y_val = val_data.iloc[:, -1]

# -----------------------------
# 3) Ensure feature columns match
# -----------------------------
if list(X_train.columns) != list(X_val.columns):
    missing_in_val = set(X_train.columns) - set(X_val.columns)
    extra_in_val   = set(X_val.columns) - set(X_train.columns)
    raise ValueError(
        "Train/Validation feature columns do not match!\n"
        f"Missing in validation: {missing_in_val}\n"
        f"Extra in validation: {extra_in_val}\n"
        "Fix by using the same columns/order in both files."
    )

# -----------------------------
# 4) Detect categorical columns (strings/categories)
# CatBoost needs indices of categorical columns
# -----------------------------
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_cols]

print(f"Detected {len(cat_cols)} categorical columns: {cat_cols}")

# -----------------------------
# 5) Initialize model
# -----------------------------
model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=RANDOM_SEED,
    verbose=200
)

# -----------------------------
# 6) Train with evaluation on validation set
# Use early stopping to prevent overfitting
# -----------------------------
model.fit(
    X_train, y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=150
)

# -----------------------------
# 7) Validate
# -----------------------------
pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, pred))
mae  = mean_absolute_error(y_val, pred)
r2   = r2_score(y_val, pred)

print("\n✅ Validation Results (Separate Validation File)")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")

# -----------------------------
# 8) Save model
# -----------------------------
joblib.dump(model, MODEL_OUT_PKL)
model.save_model(MODEL_OUT_CBM)

print(f"\n✅ Model saved to: {MODEL_OUT_PKL} and {MODEL_OUT_CBM}")