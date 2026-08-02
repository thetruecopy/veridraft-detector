import re
import numpy as np
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional imports for file processing
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

# 1. Page Configuration
st.set_page_config(
    page_title="Veridraft AI Detector Pro",
    page_icon="🔍",
    layout="wide"
)

# 2. Cache & Load Model
@st.cache_resource
def load_model():
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 3. Text Processing & Metric Helpers
def extract_text_from_file(uploaded_file):
    """Extracts raw text from TXT, PDF, or DOCX files."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = ""
    
    if file_type == "txt":
        text = uploaded_file.read().decode("utf-8")
    elif file_type == "pdf":
        if pypdf:
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            st.error("pypdf is not installed. Please add 'pypdf' to requirements.txt")
    elif file_type == "docx":
        if docx:
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            st.error("python-docx is not installed. Please add 'python-docx' to requirements.txt")
            
    # Normalize broken line breaks and extra whitespace
    normalized_text = re.sub(r'\s+', ' ', text).strip()
    return normalized_text

def calculate_burstiness(text):
    """Calculates Coefficient of Variation (CV) for sentence length variance."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) <= 1:
        return 0.0, sentences
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0, sentences
    cv = float(np.std(lengths) / mean_len)
    return cv, sentences

def log_edge_case(actual_label, notes, raw_text, score, cv_metric):
    """Posts logging payload to Discord via Webhook."""
    webhook_url = st.secrets.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        st.error("Missing DISCORD_WEBHOOK_URL in Streamlit Secrets.")
        return False
    
    payload = {
        "content": (
            f"🚨 **New Edge Case Logged**\n"
            f"• **Ground Truth:** {actual_label}\n"
            f"• **Predicted AI Score:** {score:.1%}\n"
            f"• **Burstiness (CV):** {cv_metric:.3f}\n"
            f"• **User Notes:** {notes if notes else 'None'}\n"
            f"• **Text Snippet:** ```{raw_text[:300]}...
