import re
import numpy as np
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration
st.set_page_config(
    page_title="Veridraft AI Detector Pro",
    page_icon="🔍",
    layout="centered"
)

# 2. Cache & Load RoBERTa Model
@st.cache_resource
def load_model():
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 3. Sliding Window Inference & Metric Helpers
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
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) <= 1:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0
    return float(np.std(lengths) / mean_len)

def log_edge_case(actual_label, notes, raw_text, score, cv_metric):
    """Sends logged sample payload directly to Discord via Webhook."""
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
            f"• **Text Snippet:** ```{raw_text[:300]}...```"
        )
    }
    try:
        res = requests.post(webhook_url, json=payload)
        return res.status_code in (200, 204)
    except Exception as e:
        st.error(f"Failed to post to Discord: {e}")
        return False

# 4. Main UI Layout
st.title("Veridraft AI Detector Pro 🔍")
st.write("Hybrid context & burstiness scoring pipeline for AI text verification.")

user_text = st.text_area(
    "Paste text to analyze:", 
    height=200, 
    placeholder="Enter essay, article, or document text here..."
)

if st.button("Analyze Text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Executing sliding-window chunking & burstiness analysis..."):
            base_ai_prob = chunked_ai_predict(user_text, tokenizer, model)
            burstiness_cv = calculate_burstiness(user_text)

            # Calibrated Hybrid Weighting: AI models have distinct low sentence variance (CV < 0.40)
            calibrated_prob = base_ai_prob
            if burstiness_cv < 0.40 and base_ai_prob > 0.05:
                calibrated_prob = min(1.0, base_ai_prob * 1.45)

            st.session_state["last_analysis"] = {
                "text": user_text,
                "ai_prob": calibrated_prob,
                "burstiness_cv": burstiness_cv
            }

# 5. Display Analysis Results & Feedback Form
if "last_analysis" in st.session_state:
    res = st.session_state["last_analysis"]
    st.divider()
    st.subheader("Analysis Results")
    
    col1, col2 = st.columns(2)
    col1.metric("AI Probability", f"{res['ai_prob']:.1%}")
    col2.metric("Burstiness (CV)", f"{res['burstiness_cv']:.3f}")

    if res['ai_prob'] >= 0.55:
        st.error("High probability of AI-generated text.")
    elif res['ai_prob'] >= 0.30:
        st.warning("Mixed signals / potentially AI-assisted or edited text.")
    else:
        st.success("High probability of human-written text.")

    # In-App Feedback Form
    st.divider()
    with st.expander("⚠️ Report Inaccurate Result / Log Edge Case"):
        with st.form("feedback_form", clear_on_submit=True):
            actual_label = st.radio(
                "What is the actual origin of this text?",
                ["Human Written", "AI Generated", "Mixed / Lightly Edited"]
            )
            user_notes = st.text_area("Additional context (e.g., non-native author, academic format):")
            
            if st.form_submit_button("Submit Feedback"):
                success = log_edge_case(
                    actual_label, 
                    user_notes, 
                    res["text"], 
                    res["ai_prob"], 
                    res["burstiness_cv"]
                )
                if success:
                    st.success("Logged! Case sent directly to your Discord webhook.")
