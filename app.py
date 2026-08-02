import re
import time
import io
import csv
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

# 2. Cache & Load Multi-Model Ensemble Checkpoints
@st.cache_resource
def load_ensemble_models():
    """Loads a dual-model ensemble for cross-model verification."""
    # Model 1: Primary ChatGPT RoBERTa Detector
    m1_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    tok1 = AutoTokenizer.from_pretrained(m1_name)
    mod1 = AutoModelForSequenceClassification.from_pretrained(m1_name)

    # Model 2: Secondary OpenAI RoBERTa Detector
    m2_name = "roberta-base-openai-detector"
    tok2 = AutoTokenizer.from_pretrained(m2_name)
    mod2 = AutoModelForSequenceClassification.from_pretrained(m2_name)

    return (tok1, mod1), (tok2, mod2)

(tok1, model1), (tok2, model2) = load_ensemble_models()

# 3. Detection Engine & Helper Functions
def predict_single_model(text, tokenizer, model, chunk_size=512, overlap=128):
    """Evaluates text across overlapping token windows for a single model checkpoint."""
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"][0]
    total_tokens = len(input_ids)
    
    if total_tokens <= chunk_size:
        inputs = tokenizer(text, return_tensors="pt", max_length=chunk_size, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Default index 1 represents positive AI/fake class in these architectures
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

def ensemble_ai_predict(text, weight_m1=0.60, weight_m2=0.40):
    """Combines prediction outputs from multiple models using a weighted average."""
    score_m1 = predict_single_model(text, tok1, model1)
    score_m2 = predict_single_model(text, tok2, model2)
    
    combined_score = (score_m1 * weight_m1) + (score_m2 * weight_m2)
    return combined_score, score_m1, score_m2

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

def generate_sentence_highlights(sentences):
    """Evaluates individual sentences using model ensemble for visual breakdown."""
    highlighted_html = ""
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) < 4:
            highlighted_html += f'<span style="margin-right: 4px;">{sentence}</span>'
            continue

        sentence_score, _, _ = ensemble_ai_predict(sentence)

        if sentence_score >= 0.65:
            style = "background-color: #ffcdd2; color: #b71c1c; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"
        elif sentence_score >= 0.35:
            style = "background-color: #fff9c4; color: #f57f17; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"
        else:
            style = "background-color: #c8e6c9; color: #1b5e20; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"

        highlighted_html += f'<span style="{style}" title="Ensemble AI Risk: {sentence_score:.1%}">{sentence}</span>'

    return f'<div style="line-height: 1.8; font-size: 15px; padding: 15px; background: #fafafa; border-radius: 8px; border: 1px solid #e0e0e0;">{highlighted_html}</div>'

def generate_csv_report(res):
    """Generates a structured CSV audit file string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Metric / Property", "Value"])
    writer.writerow(["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())])
    writer.writerow(["Ensemble AI Score", f"{res['ai_prob']:.1%}"])
    writer.writerow(["Primary RoBERTa Score", f"{res['score_m1']:.1%}"])
    writer.writerow(["Secondary OpenAI Score", f"{res['score_m2']:.1%}"])
    writer.writerow(["Burstiness (CV)", f"{res['burstiness_cv']:.3f}"])
    writer.writerow(["Word Count", res['word_count']])
    writer.writerow(["Sentence Count", res['sentence_count']])
    writer.writerow(["Raw Text Sample", res['text'][:300] + "..."])
    return output.getvalue()

def generate_text_report(res):
    """Generates a plain-text verification certificate/audit report."""
    lines = [
        "==================================================",
        "     VERIDRAFT AI DETECTOR PRO - AUDIT REPORT     ",
        "==================================================",
        f"Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Word Count:             {res['word_count']}",
        f"Sentences:              {res['sentence_count']}",
        "--------------------------------------------------",
        "ENSEMBLE EVALUATION BREAKDOWN:",
        f"• Final Combined Score: {res['ai_prob']:.1%}",
        f"• Primary RoBERTa:      {res['score_m1']:.1%}",
        f"• Secondary Detector:   {res['score_m2']:.1%}",
        f"• Burstiness Metric CV: {res['burstiness_cv']:.3f}",
        "--------------------------------------------------",
        "TEXT SNIPPET EVALUATED:",
        f"{res['text'][:400]}...",
        "=================================================="
    ]
    return "\n".join(lines)

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
    """Logs edge case payload to Supabase Database REST API and/or Discord Webhook."""
    discord_url = st.secrets.get("DISCORD_WEBHOOK_URL")
    supabase_url = st.secrets.get("SUPABASE_URL")
    supabase_key = st.secrets.get("SUPABASE_KEY")
    
    snippet = raw_text[:300].replace("`", "")
    note_text = notes if notes else "None"
    logged_anywhere = False

    # 1. Supabase REST Logging
    if supabase_url and supabase_key:
        try:
            endpoint = f"{supabase_url.rstrip('/')}/rest/v1/edge_cases"
            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            data = {
                "actual_label": actual_label,
                "predicted_score": float(score),
                "burstiness_cv": float(cv_metric),
                "user_notes": note_text,
                "text_snippet": snippet
            }
            sb_res = requests.post(endpoint, json=data, headers=headers)
            if sb_res.status_code in (200, 201):
                logged_anywhere = True
        except Exception as e:
            st.error(f"Supabase logging failed: {e}")

    # 2. Discord Webhook Logging
    if discord_url:
        lines = [
            "🚨 **New Edge Case Logged**",
            f"• **Ground Truth:** {actual_label}",
            f"• **Ensemble AI Score:** {score:.1%}",
            f"• **Burstiness (CV):** {cv_metric:.3f}",
            f"• **User Notes:** {note_text}",
            "• **Text Snippet:**",
            "```",
            f"{snippet}...",
            "
