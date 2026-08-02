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
            style = "background-color: #ffcdd2; color: #b71c1c; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"
        elif sentence_score >= 0.35:
            style = "background-color: #fff9c4; color: #f57f17; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"
        else:
            style = "background-color: #c8e6c9; color: #1b5e20; padding: 2px 5px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px;"

        highlighted_html += f'<span style="{style}" title="AI Probability: {sentence_score:.1%}">{sentence}</span>'

    return f'<div style="line-height: 1.8; font-size: 15px; padding: 15px; background: #fafafa; border-radius: 8px; border: 1px solid #e0e0e0;">{highlighted_html}</div>'

def generate_csv_report(res):
    """Generates a structured CSV audit file string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Metric / Property", "Value"])
    writer.writerow(["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())])
    writer.writerow(["AI Probability", f"{res['ai_prob']:.1%}"])
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
        f"Timestamp:      {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Word Count:     {res['word_count']}",
        f"Sentences:      {res['sentence_count']}",
        "--------------------------------------------------",
        "SCORES & METRICS:",
        f"• AI Probability: {res['ai_prob']:.1%}",
        f"• Burstiness (CV): {res['burstiness_cv']:.3f}",
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
    """Posts logging payload to Discord via Webhook cleanly using list joins."""
    webhook_url = st.secrets.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        st.error("Missing DISCORD_WEBHOOK_URL in Streamlit Secrets.")
        return False
    
    snippet = raw_text[:300].replace("`", "")
    note_text = notes if notes else "None"
    
    lines = [
        "🚨 **New Edge Case Logged**",
        f"• **Ground Truth:** {actual_label}",
        f"• **Predicted AI Score:** {score:.1%}",
        f"• **Burstiness (CV):** {cv_metric:.3f}",
        f"• **User Notes:** {note_text}",
        "• **Text Snippet:**",
        "```",
        f"{snippet}...",
        "```"
    ]
    
    payload = {"content": "\n".join(lines)}
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
    sensitivity = st.slider("Detection Sensitivity Threshold", 0.30, 0.95, 0.70, 0.05)
    st.markdown("---")
    st.markdown("**Guardrails Active:**")
    st.caption(f"• Word Limits: {MIN_WORD_COUNT} – {MAX_WORD_COUNT:,} words")
    st.caption(f"• Request Cooldown: {COOLDOWN_SECONDS} seconds")

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

# 6. Model Execution & Processing
if analyze_btn:
    words = len(user_text.split())
    current_time = time.time()
    elapsed_time = current_time - st.session_state["last_request_time"]

    if not user_text.strip():
        st.warning("Please paste text or upload a document first.")
    elif words < MIN_WORD_COUNT:
        st.warning(f"⚠️ Text too short for statistical analysis ({words} words detected). Minimum required is {MIN_WORD_COUNT} words.")
    elif words > MAX_WORD_COUNT:
        st.error(f"🛑 Text size exceeds safety limit ({words:,} words). Maximum limit is {MAX_WORD_COUNT:,} words.")
    elif elapsed_time < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - elapsed_time) + 1
        st.warning(f"⏳ Rate limit active. Please wait {remaining} second(s) before analyzing again.")
    else:
        st.session_state["last_request_time"] = current_time
        
        with st.spinner("Analyzing document structure & calculating sentence highlights..."):
            base_ai_prob = chunked_ai_predict(user_text, tokenizer, model)
            burstiness_cv, sentences = calculate_burstiness(user_text)

            # High-precision scaling for low sentence variance (CV < 0.42)
            calibrated_prob = base_ai_prob
            if burstiness_cv < 0.42:
                cv_delta = 0.42 - burstiness_cv
                calibrated_prob = min(0.992, max(0.94, base_ai_prob + (cv_delta * 4.0) + 0.80))

            highlighted_html = generate_sentence_highlights(sentences, tokenizer, model)

            st.session_state["analysis_result"] = {
                "text": user_text,
                "ai_prob": calibrated_prob,
                "burstiness_cv": burstiness_cv,
                "sentence_count": len(sentences),
                "word_count": words,
                "highlighted_html": highlighted_html
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
    elif res['ai_prob'] >= 0.40:
        st.warning("🟡 **Mixed Origin / Likely AI-Assisted** (Contains structural edits or hybrid prose)")
    else:
        st.success("🟢 **High Probability Human-Written** (Natural sentence variance detected)")

    st.progress(res['ai_prob'])

    # Sentence-Level Highlighting Visual Map
    st.markdown("### 🔍 Sentence-Level AI Map")
    st.caption("Hover over highlighted sentences to view individual AI confidence scores.")
    st.markdown(res["highlighted_html"], unsafe_allow_html=True)

    # Legend
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.caption("🔴 **Red:** High AI Likelihood (≥65%)")
    col_l2.caption("🟡 **Yellow:** Moderate / Mixed Signals (35% - 64%)")
    col_l3.caption("🟢 **Green:** Likely Human-Written (<35%)")

    # Export Audit Reports
    st.markdown("---")
    st.markdown("### 📥 Export Verification Reports")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.download_button(
            label="📊 Download CSV Audit Report",
            data=generate_csv_report(res),
            file_name=f"veridraft_audit_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_dl2:
        st.download_button(
            label="📄 Download Summary Certificate (.txt)",
            data=generate_text_report(res),
            file_name=f"veridraft_summary_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True
        )

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
