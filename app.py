import re
import time
import numpy as np
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Guardrail Configuration Constants
MIN_WORD_COUNT = 15
MAX_WORD_COUNT = 5000
COOLDOWN_SECONDS = 4

# Optional imports for document processing
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

# Initialize Session Rate Limiting Tracking
if "last_request_time" not in st.session_state:
    st.session_state["last_request_time"] = 0.0

# 2. Cache & Load Verified Model Checkpoint
@st.cache_resource
def load_model():
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 3. Detection Engine & Helper Functions
def chunked_ai_predict(text, tokenizer, model, chunk_size=512, overlap=128):
    """Evaluates long texts in overlapping token windows to prevent truncation."""
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"][0]
    total_tokens = len(input_ids)
    
    if total_tokens <= chunk_size:
        inputs = tokenizer(text, return_tensors="pt", max_length=chunk_size, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return float(probs[0][1].item())
    
    step = chunk_size - overlap
    chunk_scores = []
    
    for start_idx in range(0, total_tokens, step):
        end_idx = min(start_idx + chunk_size, total_tokens)
        chunk_ids = input_ids[start_idx:end_idx].unsqueeze(0)
        
        if chunk_ids.shape[1] < 30:
            continue
            
        attention_mask = torch.ones_like(chunk_ids)
        with torch.no_grad():
            outputs = model(input_ids=chunk_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            chunk_scores.append(probs[0][1].item())
            
    return float(np.mean(chunk_scores)) if chunk_scores else 0.0

def calculate_burstiness(text):
    """Calculates Coefficient of Variation (CV) for sentence length variance."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if len(sentences) <= 1:
        return 0.0, sentences
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0, sentences
    cv = float(np.std(lengths) / mean_len)
    return cv, sentences

def generate_sentence_highlights(sentences, tokenizer, model):
    """Evaluates individual sentences and returns styled HTML tags for visual breakdown."""
    highlighted_html = ""
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) < 4:
            highlighted_html += f'<span style="margin-right: 4px;">{sentence}</span>'
            continue

        inputs = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            sentence_score = float(probs[0][1].item())

        if sentence_score >= 0.65:
            # High AI risk - Red highlight
            style = "background-color: #ffcdd2; color: #b71c1c; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"
        elif sentence_score >= 0.35:
            # Medium/Mixed risk - Yellow highlight
            style = "background-color: #fff9c4; color: #f57f17; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"
        else:
            # Low AI risk / Human - Green highlight
            style = "background-color: #c8e6c9; color: #1b5e20; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"

        highlighted_html += f'<span style="{style}" title="AI Probability: {sentence_score:.1%}">{sentence}</span>'

    return f'<div style="line-height: 1.8; font-size: 15px; padding: 15px; background: #fafafa; border-radius: 8px; border: 1px solid #e0e0e0;">{highlighted_html}</div>'

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
            
    return re.sub(r'\s+', ' ', text).strip()

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
