import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Depression Prediction Tool",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Depression Prediction Tool")

# 1. Load model safely --------------------------------------------------
@st.cache_resource
def load_model(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

MODEL_PATH = "depression_xgboost_model.pkl"

try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# 2. Sidebar inputs -----------------------------------------------------
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose CSV or Excel file with the same columns you used for training",
    type=["csv", "xlsx"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:**\n- Use the cleaned dataset structure\n- No target column (`Depression`) in the upload")

# 3. Helper: load dataframe --------------------------------------------
def read_uploaded_file(file) -> pd.DataFrame:
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

# 4. Main logic ---------------------------------------------------------
if uploaded_file is not None:
    # Load
    try:
        df = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    st.subheader("Data preview")
    st.dataframe(df.head())

    # Optional: show columns
    with st.expander("Show columns"):
        st.write(list(df.columns))

    # Predict button
    if st.button("🚀 Predict All", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            try:
                # Make predictions
                preds = model.predict(df)
                # Some models may not have predict_proba
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df)[:, 1]
                else:
                    probs = np.zeros(len(preds))

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # 5. Show results ------------------------------------------------
        st.subheader("Results")

        depressed = int(np.sum(preds))
        not_depressed = int(len(preds) - depressed)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Depressed (1)", depressed)
        with col2:
            st.metric("Not Depressed (0)", not_depressed)
        with col3:
            # Use training accuracy only if you stored it; here show share of 1s
            st.metric("Share Depressed", f"{depressed / len(preds):.1%}")
        with col4:
            st.metric("Accuracy", f"{model.score(df, preds):.1%}")

        st.bar_chart(pd.Series(preds, name="Prediction").value_counts())

        # Attach results to dataframe
        result_df = df.copy()
        result_df["Prediction"] = preds
        result_df["Risk_Probability"] = probs

        st.subheader("Sample of predictions")
        st.dataframe(result_df.head())

        # Download button
        csv = result_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results as CSV",
            data=csv,
            file_name="depression_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("👈 Upload an Excel/CSV file in the sidebar to start.")

st.markdown("---")
st.caption("Built with ❤️ by GDG")
