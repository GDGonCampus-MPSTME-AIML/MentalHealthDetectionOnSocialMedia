# End‑to‑end pipeline for student mental‑health (depression) prediction

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 1. Load data ----------------------------------------------------------


DATA_PATH = "cleandata.csv-copy.xlsx"

df = pd.read_excel(DATA_PATH)

# Inspect and drop obvious identifier columns 
id_like_cols = [col for col in df.columns if col.lower() in ["id"]]
df = df.drop(columns=id_like_cols)

# Target column (from your sheet)
TARGET_COL = "Depression"

# Ensure target is categorical/binary (0/1)
y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

# 2. Identify column types ----------------------------------------------

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# 3. Preprocessing + model pipeline ------------------------------------

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", rf_clf),
    ]
)

# 4. Train/test split ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Hyperparameter tuning with CV -------------------------------------

param_grid = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    scoring="f1",      # focus on F1 due to mental‑health context
    n_jobs=-1,
    verbose=2,
)

print("Starting hyperparameter search...")
grid_search.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# 6. Evaluation on held‑out test set -----------------------------------

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nTest set performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
