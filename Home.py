import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from pgmpy.inference import VariableElimination

# ---------- CONFIG ----------
st.set_page_config(page_title="Image Classification Interface", layout="centered")
IMG_SIZE = (128, 128)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
h1 {
    font-size: 2.8em;
    font-weight: 600;
    margin-bottom: 0.3em;
}
h4 {
    font-size: 1.6em;
    font-weight: 600;
    margin-top: 1.2em;
    margin-bottom: 0.6em;
}
p {
    font-size: 1.15em;
    line-height: 1.6;
    margin-bottom: 0.8em;
}
.uploadedImage img {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    animation: fadeIn 0.6s ease-in-out;
    margin-top: 10px;
}
.stSpinner {
    font-size: 1.1em;
}
.stSuccess {
    animation: pulse 1s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes pulse {
    0% {transform: scale(1);}
    50% {transform: scale(1.02);}
    100% {transform: scale(1);}
}
.prediction-card {
    background-color: #ffffff;
    padding: 24px 28px;
    border-radius: 12px;
    border: 1px solid #ddd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin: 20px auto 16px auto;
    max-width: 640px;
    transition: box-shadow 0.3s ease, transform 0.2s ease;
}
.prediction-card:hover {
    box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}
.radio-group {
    max-width: 640px;
    margin: 0 auto 20px auto;
    padding: 10px 0;
}
.radio-group label {
    display: block;
    margin-bottom: 10px;
    font-size: 1.1em;
    cursor: pointer;
    transition: background-color 0.2s ease;
}
.radio-group label:hover {
    background-color: #f0f0f0;
}
.prediction-entry {
    background-color: #ffffff;
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 6px;
    font-size: 1.05em;
    font-weight: 500;
    color: #333;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: background-color 0.2s ease;
}

.prediction-entry:hover {
    background-color: #f0f0f0;
}

.prediction-entry span {
    color: #2e7d32;
    font-weight: 600;
}
.footer {
    text-align: center;
    font-size: 0.9em;
    color: #666;
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
}
.radio-label {
    font-size: 1.6em;
    font-weight: 600;
    margin-bottom: -1.6em;
}
div[data-testid="stRadio"] > div {
    font-size: 1.15em;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("Image Classification Interface")
st.markdown("<div style='font-size:1.05em;margin-bottom: 1.0em;'>This interface leverages MobileNetV2 features and a diverse set of machine learning models for accurate and efficient garbage classification."
            ,unsafe_allow_html=True)

# ---------- MODEL SELECTION ----------
model_files = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "SVM": "support_vector_machine_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "KNN": "KNeighbors_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "XGBoost": "xgboost_model.joblib",
    "AdaBoost": "adaboost_model.joblib",
    "MultiLayerPerceptron": "neural_network_model.joblib",
    "BayesianNetwork": "bayesian_network_model.joblib"
}

st.markdown("<div class='radio-label'>Select a Model</div>", unsafe_allow_html=True)
selected_model_name = st.radio("", list(model_files.keys()), key="model_radio")
selected_model_path = model_files[selected_model_name]

# ---------- LOAD MODELS ----------
classifier = load(selected_model_path)
label_encoder = load("label_encoder.joblib")
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(128, 128, 3))

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown('<div class="uploadedImage">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Classifying image..."):
        # Preprocess image
        image = image.resize(IMG_SIZE)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features
        features = feature_extractor.predict(img_array)

        # ---------- BAYESIAN NETWORK LOGIC ----------
        if selected_model_name == "BayesianNetwork":
            selector = load("selector.joblib")
            discretizer = load("discretizer.joblib")
            feature_cols = load("feature_cols.joblib")

            features_reduced = selector.transform(features)
            features_discrete = discretizer.transform(features_reduced)
            features_df = pd.DataFrame(features_discrete, columns=feature_cols).astype(int)
            for col in features_df.columns:
                features_df[col] = features_df[col].astype("category")

            inference = VariableElimination(classifier)

            def predict_bn(row):
                evidence_row = row[feature_cols].to_dict()
                query = inference.query(variables=['target'], evidence=evidence_row, show_progress=False)
                return np.argmax(query.values)

            prediction = predict_bn(features_df.iloc[0])
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            top_predictions_html = ""

        # ---------- STANDARD CLASSIFIER LOGIC ----------
        else:
            prediction = classifier.predict(features)[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            # ---------- TOP-3 CLASS PROBABILITIES ----------
            top_predictions_html = ""
            if hasattr(classifier, "predict_proba"):
                probs = classifier.predict_proba(features)[0]
                top_indices = np.argsort(probs)[::-1][:3]
                top_classes = label_encoder.inverse_transform(top_indices)
                top_scores = probs[top_indices]

                top_predictions_html += "<h4>Top Predictions</h4><ul>"
                for cls, score in zip(top_classes, top_scores):
                    top_predictions_html += f"<li>{cls}: {score:.2%}</li>"
                top_predictions_html += "</ul>"

        # ---------- RESULT CARD ----------
        st.markdown(f"""
        <div class="prediction-card">
        <h4>Prediction Summary</h4>
        <p><strong>Model:</strong> {selected_model_name}</p>
        <p><strong>Predicted Class:</strong> <span style="color:#2e7d32;">{predicted_label}</span></p>
        {top_predictions_html}
        </div>
        """, unsafe_allow_html=True)

        # ---------- PREDICTION HISTORY ----------
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append((selected_model_name, predicted_label))

else:
    st.info("Please upload an image to begin.")

# ---------- SIDEBAR HISTORY PANEL ----------
#st.sidebar.subheader("Recent Predictions")
st.sidebar.markdown("# Recent Predictions")
st.sidebar.markdown('<div class="sidebar-history">', unsafe_allow_html=True)
if "history" in st.session_state:
    for model, label in st.session_state.history[-5:][::-1]:
        st.sidebar.markdown(f"<div class='prediction-entry'><b>{model}</b>: <span>{label}</span></div>",
                            unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
Developed for academic research and demonstration purposes. Version 1.0 – October 2025.
</div>
""", unsafe_allow_html=True)