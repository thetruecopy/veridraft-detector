import re
import time
import io
import csv
import numpy as np
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# NLTK Safe Setup
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

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


def split_into_sentences(text):
    """Splits text into sentences using NLTK with a regex fallback."""
    try:
        sentences = nltk.sent_tokenize(text)
        if sentences:
            return sentences
    except Exception:
        pass
    
    # Fallback to regex splitting if NLTK tokenizer isn't loaded yet
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sentences if s.strip()]


def calculate_burstiness(text):
    """Calculates variation in sentence length (burstiness)."""
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return 0.0, sentences
    
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0, sentences
        
    cv = float(np.std(lengths) / mean_len)
    return cv, sentences


@st.cache_resource
def load_ensemble_models():
    """Loads ensemble classifiers into memory."""
    m1_name = "roberta-base-openai-detector"
    m2_name = "Hello-SimpleAI/chatgpt-detector-roberta"

    tok1 = AutoTokenizer.from_pretrained(m1_name)
    mod1 = AutoModelForSequenceClassification.from_pretrained(m1_name)

    tok2 = AutoTokenizer.from_pretrained(m2_name)
    mod2 = AutoModelForSequenceClassification.from_pretrained(m2_name)

    return (tok1, mod1), (tok2, mod2)


def predict_text(text, tok, mod):
    """Runs sequence classification on a given string."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mod(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    # Index 1 corresponds to AI probability across typical HF RoBERTa detectors
    if isinstance(probs, list):
        return probs[1] if len(probs) > 1 else probs[0]
    return float(probs)


def analyze_document(text, model_pair1, model_pair2):
    """Calculates overall AI likelihood and sentence-level scores."""
    (tok1, mod1), (tok2, mod2) = model_pair1, model_pair2

    # Global burstiness and sentence breakdown
    burstiness, sentences = calculate_burstiness(text)

    # Calculate global AI score
    score1 = predict_text(text, tok1, mod1)
    score2 = predict_text(text, tok2, mod2)
    global_ai_prob = (score1 + score2) / 2.0

    # Calculate sentence-level scores
    sentence_data = []
    for s in sentences:
        if len(s.split()) < 3:
            # Skip scoring ultra-short fragments
            sentence_data.append((s, global_ai_prob))
            continue
        p1 = predict_text(s, tok1, mod1)
        p2 = predict_text(s, tok2, mod2)
        avg_p = (p1 + p2) / 2.0
        sentence_data.append((s, avg_p))

    return global_ai_prob, burstiness, sentence_data


# --- UI Layout ---
st.set_page_config(page_title="Veridraft AI Detector Pro", page_icon="🔍", layout="wide")

st.title("🔍 Veridraft AI Detector Pro")
st.markdown("Analyze text for AI generation markers, sentence length variation, and sentence-level highlights.")

# Load models
try:
    with st.spinner("Loading AI detection models..."):
        model_pair1, model_pair2 = load_ensemble_models()
except Exception as e:
    st.error(f"Error loading models: {e}. If you just updated requirements.txt, please give Streamlit Cloud 1-2 minutes to finish rebuilding containers.")
    st.stop()

# Input Section
user_input = st.text_area("Paste your text below to analyze:", height=250)

uploaded_file = st.file_uploader("Or upload a document (.txt, .pdf, .docx):", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        user_input = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf" and pypdf:
        reader = pypdf.PdfReader(uploaded_file)
        user_input = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and docx:
        doc = docx.Document(uploaded_file)
        user_input = "\n".join([p.text for p in doc.paragraphs])

if st.button("Analyze Text", type="primary"):
    words = user_input.strip().split()
    if len(words) < MIN_WORD_COUNT:
        st.warning(f"Please enter at least {MIN_WORD_COUNT} words for reliable detection.")
    elif len(words) > MAX_WORD_COUNT:
        st.error(f"Text exceeds maximum limit of {MAX_WORD_COUNT} words.")
    else:
        with st.spinner("Analyzing text signals..."):
            ai_prob, burstiness, sentence_data = analyze_document(user_input, model_pair1, model_pair2)

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall AI Probability", f"{ai_prob * 100:.1f}%")
        with col2:
            st.metric("Burstiness (Sentence Length Variance)", f"{burstiness:.2f}")

        # Sentence Level Map
        st.subheader("🔍 Sentence-Level AI Map")
        st.caption("Hover over highlighted sentences to view individual AI confidence scores.")

        html_output = "<div style='line-height: 1.8; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
        for sent, score in sentence_data:
            pct = score * 100
            if pct >= 65:
                bg_color = "rgba(255, 99, 71, 0.3)" # Red
            elif pct >= 35:
                bg_color = "rgba(255, 215, 0, 0.3)" # Yellow
            else:
                bg_color = "rgba(144, 238, 144, 0.3)" # Green

            html_output += f"<span style='background-color: {bg_color}; margin: 2px; padding: 2px 4px; border-radius: 3px;' title='AI Score: {pct:.1f}%'>{sent}</span> "
        html_output += "</div>"

        st.markdown(html_output, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<span style='color:red;'>🔴 Red: High AI Likelihood (≥65%)</span> | "
                    "<span style='color:orange;'>🟡 Yellow: Mixed Signals (35%-64%)</span> | "
                    "<span style='color:green;'>🟢 Green: Likely Human (<35%)</span>", 
                    unsafe_allow_html=True)
