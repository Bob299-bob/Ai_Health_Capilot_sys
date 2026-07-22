import streamlit as st
st.set_page_config(
    page_title="AI Healthcare Copilot",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.main{
    background:#F4F7FE;
}

.title{
    font-size:45px;
    font-weight:bold;
    color:#0077B6;
}

.subtitle{
    font-size:20px;
    color:#555;
}

.card{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,.1);
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

from brain import chatbot,search_net,generate_report,syptom_sys,extract_pdf,rag_system,retrieve_sys,brain_sys,pneumon_sys,heart_model
import pandas as pd
import gdown
import os

@st.cache_data
def syptom_data():
    file_path = "symptom_columns.csv"
    if not os.path.exists(file_path):
        file_id = "1pJ_L_RLqzcTpAHMl54IKbtTqz4LcAxwG"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, file_path, quiet=False)

    return pd.read_csv(file_path)

data = syptom_data()
symptoms_list = data.drop("diseases", axis=1).columns.tolist()

with st.sidebar:

    st.markdown("## 🩺 AI Healthcare Copilot")
    st.caption("Select a healthcare service from the list below.")

    st.divider()

    module = st.selectbox(
        "📋 Choose a Module",
        (
            "🏠 Dashboard",
            "❤️ Heart Disease",
            "🧠 Brain Tumor",
            "🫁 Pneumonia",
            "🤒 Symptom Checker",
            "📄 Medical Report Analyzer",
            "💬 AI Chatbot"
        ),
        index=0
    )

    st.divider()

    st.markdown("### 📌 How to Use")
    st.markdown("""
    1. Select a module above.
    2. Upload the required image or report (if needed).
    3. Enter the requested information.
    4. Click **Predict** or **Analyze**.
    5. Review the AI-generated results carefully.
    """)

    st.info(
        "⚠️ AI predictions are for screening purposes only and should not replace professional medical advice."
    )

if module == "🏠 Dashboard":

    st.markdown("""
    <h1 style='text-align:center;color:#0F62FE;'>
        🩺 AI Healthcare Copilot
    </h1>
    <h4 style='text-align:center;color:gray;'>
        Your Intelligent Medical Screening & Health Assistant
    </h4>
    """, unsafe_allow_html=True)

    st.divider()

    st.info("""
    ### 👋 Welcome

    AI Healthcare Copilot is designed to assist you with:

    ❤️ Heart Disease Risk Assessment

    🧠 Brain Tumor MRI Analysis

    🫁 Pneumonia & COVID Chest X-ray Detection

    🤒 Symptom-Based Disease Prediction

    📄 AI Medical Report Analysis

    💬 AI Healthcare Chatbot
    """)

    st.divider()

    st.subheader("🏥 Healthcare Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("❤️ Heart Disease")
        st.caption("Assess heart disease risk using patient health information.")

    with col2:
        st.success("🧠 Brain Tumor")
        st.caption("Analyze MRI scans to identify possible brain tumors.")

    with col3:
        st.success("🫁 Pneumonia")
        st.caption("Detect Normal, COVID-19, or Pneumonia from Chest X-rays.")

    st.write("")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.success("🤒 Symptom Checker")
        st.caption("Predict possible diseases from selected symptoms.")

    with col5:
        st.success("📄 Medical Reports")
        st.caption("Upload medical reports and ask health-related questions.")

    with col6:
        st.success("💬 AI Chatbot")
        st.caption("Get instant answers to general healthcare questions.")

    st.divider()

    st.subheader("💡 Healthy Living Tips")

    st.markdown("""
    ✅ Drink enough water every day.

    ✅ Eat a balanced diet rich in fruits and vegetables.

    ✅ Exercise for at least 30 minutes daily.

    ✅ Sleep 7–9 hours each night.

    ✅ Avoid smoking and excessive alcohol.

    ✅ Schedule regular health check-ups.
    """)

    st.divider()

    st.subheader("🚨 Emergency Warning")

    st.error("""
If you experience **chest pain, severe difficulty breathing, sudden weakness,
loss of consciousness, or heavy bleeding**, seek immediate emergency medical care.
Do not rely solely on AI predictions in emergency situations.
""")

    st.divider()

    st.caption(
        "⚠️ This application is intended for educational and screening purposes only. "
        "It does not replace professional medical advice, diagnosis, or treatment."
    )

# Heart Disease
elif module == "❤️ Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    # Age
    age = st.number_input("Age", min_value=1, max_value=100, value=25)

    # Sex
    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
    sex = 1 if sex == "Male" else 0

    # Chest Pain Type
    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )
    cp = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }[cp]

    # Resting Blood Pressure
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=250,
        value=120
    )

    # Cholesterol
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200
    )

    # Fasting Blood Sugar
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )
    fbs = 1 if fbs == "Yes" else 0

    # Resting ECG
    restecg = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )
    restecg = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }[restecg]

    # Maximum Heart Rate
    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150
    )

    # Exercise Induced Angina
    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )
    exang = 1 if exang == "Yes" else 0

    # ST Depression
    oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    # Slope
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )
    slope = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }[slope]

    # Number of Major Vessels
    ca = st.selectbox(
        "Major Vessels Colored by Fluoroscopy",
        [0, 1, 2, 3]
    )

    # Thalassemia
    thal = st.selectbox(
        "Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ]
    )
    thal = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }[thal]

    if st.button("Predict Heart Disease"):
        input_data = [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]]
        pred=heart_model.predict(input_data)
        prob = heart_model.predict_proba(input_data)
        if pred[0] == 1:
            st.error("❤️ Heart Disease Detected")
            st.write(f"Confidence: {prob[0][1]*100:.2f}%")
        else:
            st.success("✅ No Heart Disease Detected")
            st.write(f"Confidence: {prob[0][0]*100:.2f}%")        
# Brain Tumor
elif module == "🧠 Brain Tumor":
    st.header("🧠 Brain Tumor Detection")

    image = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "png", "jpeg"]
    )

    if st.button("Predict"):
        if image is not None:        
            result,confidence=brain_sys(image)
            if confidence >= 80:
                st.success(f"Prediction: {result}")
                st.info(f"Confidence: {confidence:.2f}%")
            elif confidence >= 60:
                st.warning(
                f"Prediction: {result}\n\n"
                f"Confidence: {confidence:.2f}%\n"
                "The confidence is moderate. Please upload a clearer MRI image."
                )
            else:
                st.error(
                    f"Low confidence ({confidence:.2f}%). "
                    "Please upload a valid and clear MRI scan."
                )
        else:
            st.warning('Please Upload MRI')

# Pneumonia
elif module == "🫁 Pneumonia":
    st.header("🫁 Pneumonia Detection")

    image = st.file_uploader(
        "Upload Chest X-Ray",
        type=["jpg", "png", "jpeg"]
    )

    if st.button("Predict"):
        if image is not None:
            result,confidence=pneumon_sys(image)
            if confidence>=80:
                st.success(result)
            elif confidence>=60:
                st.warning('Please upload clear image')
            else:
                st.error('Kindly upload the chest x-ray')
        else:
            st.warning('please upload x-ray')
# Symptom Checker
elif module == "🤒 Symptom Checker":
    st.header("🤒 Symptom Checker")

    symptoms = st.multiselect(
        "Select Symptoms",
        symptoms_list
    )

    if st.button("Predict Disease"):
        result = syptom_sys(symptoms)

        st.success(f"Predicted Disease: {result}")

# Report Analyzer
elif module == "📄 Medical Report Analyzer":
    st.header("📄 Medical Report Analyzer")

    pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )
    query=st.text_input('Enter your query')
    if st.button("Analyze"):
        if pdf is not None:
            pdf_text=extract_pdf(pdf)
            index,pdf_text=rag_system(pdf_text)
            chunk=retrieve_sys(index,pdf_text,query)
            answer=search_net(query)
            result=generate_report(chunk,answer,query)
            st.write(result)

# Chatbot
elif module == "💬 AI Chatbot":
    st.header("💬 AI Health Chatbot")

    question = st.text_input("Ask your question")

    if st.button("Ask"):
        if question.strip():
            result=chatbot(question)
            st.write(result)
