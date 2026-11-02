import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="PathoClass AI", layout="wide")
st.title("🧬 PathoClass: Genomic Pathogenicity Predictor")

# Load model assets
@st.cache_resource
def load_assets():
    model = joblib.load("xgb_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

try:
    model, feature_names = load_assets()
except FileNotFoundError:
    st.error("Run train.py first to generate the model files.")
    st.stop()

# Sidebar Inputs
st.sidebar.header("Variant Parameters")
input_data = {}
# New code
for feature in feature_names:
    # This allows you to type in any number, bypassing the 1.0 limit
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.00)

input_df = pd.DataFrame([input_data])

col1, col2 = st.columns([1, 2])

# Prediction Logic
with col1:
    st.subheader("Diagnostic Prediction")
    if st.button("Predict Pathogenicity", type="primary"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        
        if pred == 1:
            st.error("### Pathogenic")
            st.write(f"Confidence: **{prob[1] * 100:.1f}%**")
        else:
            st.success("### Benign")
            st.write(f"Confidence: **{prob[0] * 100:.1f}%**")

# Interpretability Graph
with col2:
    st.subheader("AI Feature Importances")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale="Blues")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
    st.plotly_chart(fig, use_container_width=True)