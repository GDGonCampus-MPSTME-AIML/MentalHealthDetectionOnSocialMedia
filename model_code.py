import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from text_features import TextFeatureEngineer

df = pd.read_csv("Mental-Health-Twitter.csv", encoding="latin1").dropna(subset=["post_text", "label"])
X_raw = df[["post_text"]]
y = df["label"].astype(int)

# Balanced weight to improve accuracy on smaller depression class
pos_ratio = (y == 0).sum() / (y == 1).sum()

preprocess = ColumnTransformer([
    ("text", TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 3)), "post_text"),
    ("num", SimpleImputer(strategy="median"), ["polarity", "subjectivity", "i_usage", "abs_usage", "dep_word_count", "flesch_ease"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["sentiment_label"]),
])

xgb_full_pipe = Pipeline([
    ("feat", TextFeatureEngineer(text_col="post_text")),
    ("preprocess", preprocess),
    ("model", XGBClassifier(scale_pos_weight=pos_ratio, max_depth=8, learning_rate=0.05, n_estimators=500, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.15, stratify=y, random_state=42)
xgb_full_pipe.fit(X_train, y_train)
joblib.dump(xgb_full_pipe, "depression_xgboost_model_text_only.pkl")
print("Model Retrained Successfully.")