import streamlit as st
import pandas as pd
import mlflow.pyfunc
import mlflow
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv(".env_streamlit")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://orange-space-halibut-j74q7x57p9x25gv4-5000.app.github.dev")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "HeartDisease")
mlflow.set_tracking_uri(TRACKING_URI)
if "pred" not in st.session_state:
    st.session_state["pred"] = None

@st.cache_resource(show_spinner="Loading model...")
def load_model():

    model_name = MODEL_NAME
    stage = "Production"
    model_uri = f"models:/{model_name}/{stage}"
    
    try:
        import mlflow.sklearn   
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded from MLflow Model Registry")
        return model
    except:
        from joblib import dump, load
        model_path = os.path.join("model", "model.joblib")
        model = load(model_path)
        print("Model loaded from local joblib file")
        return model

def make_prediction(model):
    age = st.session_state["age"]
    sex_str = st.session_state["gender"]
    cp_str = st.session_state["cp"]
    trestbps = st.session_state["trestbps"]
    chol = st.session_state["chol"]
    fbs = st.session_state["fbs"]         
    restecg = st.session_state["restecg"]
    thalach = st.session_state["thalach"]
    exang_str = st.session_state["exang"] 
    oldpeak = st.session_state["oldpeak"]
    slope = st.session_state["slope"]
    ca = st.session_state["ca"]
    thal = st.session_state["thal"]

    sex = 1 if sex_str == "Male" else 0

    exang = 1 if exang_str == "Yes" else 0   
    fbs = 1 if fbs else 0                  
    slope_map = {
    "Upsloping (1)": 1,
    "Flat (2)": 2,
    "Downsloping (3)": 3
}
    slope = slope_map[slope]

    thal_map = {
    "Normal (3)": 3,
    "Fixed defect (6)": 6,
    "Reversible defect (7)": 7
}
    thal = thal_map[thal]

    cp_map = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4
}
    cp = cp_map[cp_str]

    restecg_map = {
    "Normal (0)": 0,
    "ST-T abnormality (1)": 1,
    "Left ventricular hypertrophy (2)": 2
}
    restecg = restecg_map[restecg]


    X_pred = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })


    pred_proba = model.predict_proba(X_pred)[0]
    pred_class = int(pred_proba[1] > 0.5)  
    risk_probability = round(pred_proba[1] * 100, 2)

    st.session_state["pred"] = (pred_class, risk_probability)

if __name__ == "__main__":
    st.title("Heart Disease Risk Prediction")

    model = load_model()

    st.header("Enter Patient Information")

    with st.form(key="form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.number_input("Age", min_value=1, max_value=120, value=50, step=1, key="age")
            st.selectbox("Gender", options=["Male", "Female"], index=0, key="gender")
            st.selectbox("Chest Pain Type (cp)", 
                         options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
                         key="cp")

        with col2:
            st.number_input("Resting Blood Pressure (trestbps, mm Hg)", min_value=50, max_value=250, value=120, step=1, key="trestbps")
            st.number_input("Serum Cholesterol (chol, mg/dl)", min_value=100, max_value=600, value=200, step=1, key="chol")
            st.checkbox("Fasting Blood Sugar > 120 mg/dl", key="fbs")

        with col3:
            st.selectbox("Resting ECG Results (restecg)", 
                         options=["Normal (0)", "ST-T abnormality (1)", "Left ventricular hypertrophy (2)"],
                         format_func=lambda x: x, key="restecg")
            st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150, step=1, key="thalach")
            st.selectbox("Exercise Induced Angina (exang)", options=["No", "Yes"], key="exang")

        # Additional inputs in a second row
        st.markdown("### Additional Parameters")
        col4, col5, col6 = st.columns(3)
        with col4:
            st.slider("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1, key="oldpeak")
        with col5:
            st.selectbox("Slope of Peak Exercise ST Segment", 
                         options=["Upsloping (1)", "Flat (2)", "Downsloping (3)"],
                         format_func=lambda x: x.split()[0], key="slope")
        with col6:
            st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0, step=1, key="ca")
            st.selectbox("Thalassemia", 
                         options=["Normal (3)", "Fixed defect (6)", "Reversible defect (7)"],
                         format_func=lambda x: x, key="thal")

        submitted = st.form_submit_button("Calculate Risk", type="primary", on_click=make_prediction, kwargs=dict(model=model))

    if st.session_state["pred"] is not None:
        pred_class, risk_probability = st.session_state["pred"]
        if pred_class == 1:
            st.error(f"⚠️ High Risk: The model predicts presence of heart disease (probability: {risk_probability}%)")
        else:
            st.success(f"✅ Low Risk: The model predicts no heart disease (probability of disease: {risk_probability}%)")
    else:
        st.info("Please fill in the information above and click 'Calculate Risk' to get a prediction.")

    st.caption("This is a demonstration app using the UCI Heart Disease dataset. It is not a substitute for professional medical advice.")