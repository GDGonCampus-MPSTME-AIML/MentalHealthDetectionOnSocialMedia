# mental_health_rf_and_xgb.py
# Train & compare Random Forest and XGBoost for depression prediction


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
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import pickle
# added
#import h5py
#from tensorflow import keras
#from keras.models import load_model


# 1. Load data ----------------------------------------------------------


DATA_PATH = "C:\\AYUSH\\GDG\\cleandata.xlsx"
df = pd.read_excel(DATA_PATH)


# Drop ID-like columns
id_like_cols = [col for col in df.columns if col.lower() == "id"]
df = df.drop(columns=id_like_cols)


TARGET_COL = "Depression"


y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])
X = X.dropna(axis=1, how="all")


# 2. Column types -------------------------------------------------------


categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()


# Fix mixed data types: make all categorical columns strings
for col in categorical_cols:
    X[col] = X[col].astype(str)


# 3. Preprocessing with imputation -------------------------------------


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)


preprocess = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols),
    ]
)


# 4. Train/test split --------------------------------------------------


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# 5. Random Forest Pipeline --------------------------------------------


rf_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


rf_param_grid = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
}


rf_grid = GridSearchCV(
    rf_pipe,
    rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)


rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_


# 6. XGBoost Pipeline --------------------------------------------------


xgb_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "model",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


xgb_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}


xgb_grid = GridSearchCV(
    xgb_pipe,
    xgb_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)


xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_


# 7. Evaluation function -----------------------------------------------


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} RESULTS")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# 8. Compare models ----------------------------------------------------


evaluate(rf_best, X_test, y_test, "Random Forest")
evaluate(xgb_best, X_test, y_test, "XGBoost")


# 9. Save best XGBoost model ------------------------------------------


joblib.dump(xgb_best, "depression_xgboost_model.pkl")
print("\nSaved XGBoost pipeline as 'depression_xgboost_model.pkl'")


#with h5py.File("depression_xgboost_model.h5", "w") as f:
#    f.create_dataset("model", data=pickle.dumps(xgb_best))
#model = xgb_best
# Save model
#model.save('my_model.h5')

# Load model (h5py is used internally by Keras)
#model = load_model('my_model.h5')
#print("\nSaved XGBoost pipeline as 'depression_xgboost_model.h5'")


