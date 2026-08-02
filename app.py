import re
import numpy as np
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) <= 1:
        return 0.0, sentences
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0, sentences
    cv = float(np.std(lengths) / mean_len)
    return cv, sentences

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
            f"• **Text Snippet:** ```{raw_text[:300]}...```"
        )
    }
    try:
        res = requests.post(webhook_url, json=payload)
        return res.status_code in (200, 204)
    except Exception as e:
        st.error(f"Failed to post to Discord: {e}")
        return False

# 4. Sidebar Controls
with st.sidebar:
    st.title("⚙️ Model Controls")
    st.markdown("**Veridraft Engine:** RoBERTa Multi-Window")
    sensitivity = st.slider("Detection Sensitivity Threshold", 0.30, 0.90, 0.50, 0.05)
    st.markdown("---")
    st.markdown("**Metrics Explanation:**")
    st.markdown("- **AI Probability:** Multi-window token analysis score.")
    st.markdown("- **Burstiness (CV):** Sentence length variance (values < 0.42 indicate uniform AI-like rhythm).")

# 5. Main UI Header & Inputs
st.title("Veridraft AI Detector Pro 🔍")
st.caption("Hybrid macro-context chunking & burstiness scoring pipeline")

tab_text, tab_file = st.tabs(["📝 Paste Text", "📁 Upload Document"])

user_text = ""

with tab_text:
    pasted_text = st.text_area(
        "Paste document text:", 
        height=220, 
        placeholder="Paste essay, research paper, or article here..."
    )
    if pasted_text:
        user_text = pasted_text

with tab_file:
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf, .docx, .txt)", 
        type=["pdf", "docx", "txt"]
    )
    if uploaded_file:
        user_text = extract_text_from_file(uploaded_file)
        if user_text:
            st.info(f"Extracted {len(user_text.split())} words from {uploaded_file.name}")

analyze_btn = st.button("Run Hybrid Analysis", type="primary", use_container_width=True)

# 6. Model Execution & Calibrated Hybrid Scoring
if analyze_btn:
    if not user_text.strip():
        st.warning("Please paste text or upload a document first.")
    else:
        with st.spinner("Executing sliding-window chunking & burstiness analysis..."):
            base_ai_prob = chunked_ai_predict(user_text, tokenizer, model)
            burstiness_cv, sentences = calculate_burstiness(user_text)

            # Aggressive probability scaling for low sentence variance (CV < 0.42)
            calibrated_prob = base_ai_prob
            if burstiness_cv < 0.42:
                cv_delta = 0.42 - burstiness_cv
                calibrated_prob = min(0.98, max(0.75, base_ai_prob + (cv_delta * 3.0) + 0.50))

            st.session_state["analysis_result"] = {
                "text": user_text,
                "ai_prob": calibrated_prob,
                "burstiness_cv": burstiness_cv,
                "sentence_count": len(sentences),
                "word_count": len(user_text.split())
            }

# 7. Render Full Results Dashboard
if "analysis_result" in st.session_state:
    res = st.session_state["analysis_result"]
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AI Probability", f"{res['ai_prob']:.1%}")
    col2.metric("Burstiness (CV)", f"{res['burstiness_cv']:.3f}")
    col3.metric("Word Count", res['word_count'])
    col4.metric("Sentences", res['sentence_count'])

    # Verdict Banner
    if res['ai_prob'] >= sensitivity:
        st.error(f"🔴 **High Probability AI-Generated** (Exceeds {sensitivity:.0%} sensitivity threshold)")
    elif res['ai_prob'] >= 0.30:
        st.warning("🟡 **Mixed Origin / Likely AI-Assisted** (Contains structural edits or hybrid prose)")
    else:
        st.success("🟢 **High Probability Human-Written** (Natural sentence variance detected)")

    st.progress(res['ai_prob'])

    # Feedback Form Section
    st.markdown("---")
    with st.expander("⚠️ Report Inaccurate Result / Log Edge Case"):
        with st.form("feedback_form", clear_on_submit=True):
            actual_label = st.radio(
                "What is the actual origin of this text?",
                ["Human Written", "AI Generated", "Mixed / Lightly Edited"]
            )
            user_notes = st.text_area("Additional context (e.g., non-native writer, academic format):")
            
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
