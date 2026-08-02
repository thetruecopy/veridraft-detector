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

# 3. Helper Functions
def calculate_burstiness(text):
    """Calculates Coefficient of Variation (CV) for sentence lengths."""
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
        with st.spinner("Processing text and calculating metrics..."):
            # Model Inference
            inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                ai_prob = probs[0][1].item()  # Probability of AI/Fake

            # Feature Calculation
            burstiness_cv = calculate_burstiness(user_text)

            # Store in session state for stateful feedback handling
            st.session_state["last_analysis"] = {
                "text": user_text,
                "ai_prob": ai_prob,
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

    if res['ai_prob'] >= 0.70:
        st.error("High probability of AI-generated text.")
    elif res['ai_prob'] >= 0.40:
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
            user_notes = st.text_area("Additional context (e.g., non-native writer, technical document):")
            
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
