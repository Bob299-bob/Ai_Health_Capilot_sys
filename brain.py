import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import joblib
import faiss
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array,load_img
import numpy as np
import pandas as pd
from ddgs import DDGS
from groq import Groq
from dotenv import load_dotenv
import os 
import streamlit as st

#brain model is to heavy so we use this 
@st.cache_resource
def load_brain_model():
    import gdown

    model_path = "bmodel.h5"

    if not os.path.exists(model_path):
        file_id = "1xPv-pkYRntD_BzFgTpq0K9yQq24qPo9X"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            model_path,
            quiet=False
        )

    return load_model(model_path)


@st.cache_resource
def load_pneumonia_model():
    import gdown

    model_path = "model.h5"

    if not os.path.exists(model_path):
        file_id = "14IyMX994U8WAREvVEX2HFwv0gTjrh8NE"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            model_path,
            quiet=False
        )

    return load_model(model_path)
try:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets."
        )

    client = Groq(api_key=api_key)
    print("Groq Loaded Successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
    raise e
#chunking
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
#Vector Embedding
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedder()
#load model
@st.cache_resource
def load_ml_models():
    heart = joblib.load("models/heart.pkl")
    symptom = joblib.load("models/syptom.pkl")
    return heart, symptom

heart_model, syptom_model = load_ml_models()

#extract  pdf data
def extract_pdf(data_path):
    extract_data=''
    with pdfplumber.open(data_path) as pdf:
        for page in pdf.pages:
            page_text=page.extract_text()
            if page_text:
                extract_data+=page_text+'\n'
    return extract_data

#create rag system 
def rag_system(data):
    pdf_text=splitter.split_text(data)
    pdf_embedding=model.encode(pdf_text).astype('float32')
    faiss.normalize_L2(pdf_embedding)
    index=faiss.IndexFlatIP(pdf_embedding.shape[1])
    index.add(pdf_embedding)
    return index,pdf_text

#create retrieve system
def retrieve_sys(index,pdf_text,query):
    chunks=[]
    #vector embed for query
    query_embed=model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embed)
    #find indices
    distance,indices=index.search(query_embed,k=min(5,len(pdf_text)))
    for idx in indices[0]:
        chunks.append(pdf_text[idx])
    return chunks

def search_net(query):
    results=[]
    with DDGS() as ddgs:
        search_result=list(ddgs.text(query,max_results=5))
        for result in search_result:
            results.append(f'''
                        Content:{result.get('body','')}
                           ''')
    return '\n\n'.join(results)

def generate_report(answer,result,query):
    context="\n\n".join(answer)
    context=context[:6000]
    prompt = f"""
You are an experienced Medical Report Analysis Assistant.

Patient Medical Report:
{context}

Additional Information from Web Search:
{result}

Task:
Analyze the medical report and generate a detailed patient-friendly report.

Rules:
- Use ONLY the information provided in the medical report.
- Do NOT make up diagnoses.
- Clearly mention if any values are outside normal ranges.
- Explain medical terms in simple language.
- Mention possible health concerns only if supported by the report.
- If information is insufficient, explicitly state that.
- Avoid giving definitive medical diagnoses.
- Suggest consulting a qualified doctor when appropriate.

Generate the report in the following format:

# Medical Report Summary

## Patient Overview
- Brief summary of the report

## Key Test Results
- List important findings
- Mention normal and abnormal values

## Abnormal Findings
- Highlight any concerning results
- Explain what they may indicate

## Health Interpretation
- Simple explanation of what the report suggests

## Risk Indicators
- Any potential risks visible in the report

## Recommended Follow-Up
- Suggested medical consultations
- Additional tests if necessary

## Conclusion
- Overall assessment in simple language

User Question:
{query}
"""
    response=client.chat.completions.create(
         model='llama-3.1-8b-instant',
        messages=[{'role':'user','content':prompt}]
    )
    return response.choices[0].message.content

#ai chatbot
def chatbot(query):
    prompt=f'''
You are MedAssist Pro, an AI-powered Healthcare Assistant.

Your responsibilities:
- Answer health-related questions in simple and easy-to-understand language.
- Explain diseases, symptoms, causes, risk factors, prevention, diagnosis, and treatments.
- Suggest healthy lifestyle changes when appropriate.
- Recommend consulting a qualified doctor for serious or emergency symptoms.
- Do NOT prescribe controlled medications or provide a final medical diagnosis.
- If the user describes emergency symptoms (such as chest pain, severe bleeding, difficulty breathing, stroke symptoms, or loss of consciousness), advise them to seek immediate emergency medical care.
- If the information is insufficient, ask follow-up questions before giving suggestions.
- Keep responses accurate, concise, and medically responsible.

Response format:
1. Possible Explanation
2. Common Causes
3. Recommended Next Steps
4. Home Care Tips (if appropriate)
5. When to See a Doctor
6. Disclaimer

User Question:
{query}
'''
    response=client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role':'user','content':prompt}]
    )
    return response.choices[0].message.content

#brain model
def brain_sys(data):
    #to load img
    img=load_img(data,target_size=(200,200))
    #convert into array
    img_array=img_to_array(img)
    #expand dims
    expand_dim=np.expand_dims(img_array,axis=0)
    #resize img
    fin_img=expand_dim/255.0
    brain_model=load_brain_model()
    pred=brain_model.predict(fin_img)
    class_names=['glioma','meningioma','notumor','pituitary','unknown']
    result=class_names[np.argmax(pred)]
    confidence=np.max(pred)*100
    return result,confidence

#pneumonia model
def pneumon_sys(data):
    img=load_img(data,target_size=(200,200))
    img_arr=img_to_array(img)
    exp_img=np.expand_dims(img_arr,axis=0)
    fin=exp_img/255.0
    pnemon_model=load_pneumonia_model()
    pred=pnemon_model.predict(fin)
    class_names=['Covid','Normal','Pneumonia']
    result=class_names[np.argmax(pred)]
    confidence=np.max(pred)*100
    return result,confidence

@st.cache_data
def syptom_data():
    import gdown
    file_path = "syptom.csv"

    if not os.path.exists(file_path):
        file_id = "1vhMGnVqpMplEiPngjNuA8AdUW_rZI35u"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, file_path, quiet=False)

    return pd.read_csv(file_path)

def syptom_sys(select):
    # Load dataset only to get column names
    data=syptom_data()
    # Feature names
    symptoms = data.drop("diseases", axis=1).columns
    # Initialize all symptoms as 0
    user = {}
    for col in symptoms:
        user[col]=0

    for symptom in select:
        user[symptom]=1
    
    # Convert to DataFrame
    input_df = pd.DataFrame([user])
    # Predict
    prediction = syptom_model.predict(input_df)
    print('result\n',prediction[0])
    return prediction[0]
