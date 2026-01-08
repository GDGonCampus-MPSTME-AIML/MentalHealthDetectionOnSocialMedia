import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from text_features import TextFeatureEngineer

st.set_page_config(page_title="Depression Checker Pro", page_icon="🧠")
st.title("🧠 Depression Text Checker")

@st.cache_resource
def load_model():
    return joblib.load("depression_xgboost_model_text_only.pkl")

try:
    model = load_model()
except:
    st.error("Model file not found. Ensure app2.py was run successfully.")
    st.stop()

user_text = st.text_area("Analyze your thoughts:", placeholder="Type here...", height=150)

if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    if len(user_text.strip()) < 3:
        st.warning("Please provide more text.")
    else:
        input_df = pd.DataFrame({"post_text": [user_text]})
        
        eng = TextFeatureEngineer()
        feat = eng.transform(input_df)
        
        # 1. Get Base Probability from ML Model
        base_proba = model.predict_proba(input_df)[0, 1]
        
        # 2. SEVERITY OVERRIDE LOGIC
        emergency_list = feat['emergency_words'].iloc[0]
        general_risk_list = feat['risk_words'].iloc[0]
        
        final_proba = base_proba
        
        # Rule 1: Immediate jump for extreme language
        if len(emergency_list) > 0:
            final_proba = max(0.95, base_proba) # Set to at least 95%
        
        # Rule 2: Boost for general risk words
        elif len(general_risk_list) > 0:
            final_proba = min(0.85, base_proba + (len(general_risk_list) * 0.10))

        # 3. Display Results
        st.subheader("Analysis Result")
        
        if final_proba >= 0.80:
            st.error(f"**High Probability of Depression Markers:** {final_proba:.1%}")
            if len(emergency_list) > 0:
                st.markdown("⚠️ **Critical Indicator Found:** The model detected high-severity language.")
        elif final_proba >= 0.45:
            st.warning(f"**Moderate Indicators Detected:** {final_proba:.1%}")
        else:
            st.success(f"**No Significant Indications Detected:** {final_proba:.1%}")

        # Highlight detected words
        all_detected = list(set(emergency_list + general_risk_list))
        if all_detected:
            st.markdown(f"**Detected Markers:** {', '.join([f'`{w}`' for w in all_detected])}")

        # Visuals
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", f"{feat['polarity'].iloc[0]:.2f}")
        c2.metric("Self-Focus", int(feat['i_usage'].iloc[0]))
        c3.metric("Markers Found", len(all_detected))

        wc = WordCloud(background_color="white", width=800, height=400).generate(user_text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

st.sidebar.info("This app identifies linguistic patterns often associated with depression using a combination of XGBoost and priority keyword overrides.")