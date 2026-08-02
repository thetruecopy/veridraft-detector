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
    m1_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    tok1 = AutoTokenizer.from_pretrained(m1_name)
    mod1 = AutoModelForSequenceClassification.from_pretrained(m1_name)

    m2_name = "roberta-base-openai-detector"
    tok2 = AutoTokenizer.from_pretrained(m2_name)
    mod2 = AutoModelForSequenceClassification.from_pretrained(m2_name)

    return (tok1, mod1), (tok2, mod2)

(tok1, model1), (tok2, model2) = load_ensemble_models()

# 3. Detection Engine & Helper Functions
def predict_single_model(text, tokenizer, model, chunk_size=512, overlap=128):
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

def ensemble_ai_predict(text, weight_m1=0.60, weight_m2=0.40):
    score_m1 = predict_single_model(text, tok1, model1)
    score_m2 = predict_single_model(text, tok2, model2)
    
    combined_score = (score_m1 * weight_m1) + (score_m2 * weight_m2)
    return combined_score, score_m1, score_m2

def calculate_burstiness(text):
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
    time_str = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    snippet_str = res['text'][:400] + "..."
    
    lines = [
        "==================================================",
        "     VERIDRAFT AI DETECTOR PRO - AUDIT REPORT     ",
        "==================================================",
        f"Timestamp:              {time_str}",
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
        snippet_str,
        "=================================================="
    ]
    return "\n".join(lines)

def extract_text_from_file(uploaded_file):
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
    discord_url = st.secrets.get("DISCORD_WEBHOOK_URL")
    supabase_url = st.secrets.get("SUPABASE_URL")
    supabase_key = st.secrets.get("SUPABASE_KEY")
    
    clean_snippet = raw_text[:300].replace('\n', ' ')
    note_text = notes.strip() if notes else "None"
    score_formatted = f"{score:.1%}"
    cv_formatted = f"{cv_metric:.3f}"
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
                "text_snippet": clean_snippet
            }
            sb_res = requests.post(endpoint, json=data, headers=headers)
            if sb_res.status_code in (200, 201):
                logged_anywhere = True
        except Exception as e:
            st.error(f"Supabase logging failed: {e}")

    # 2. Discord Webhook Logging
    if discord_url:
        discord_lines = [
            "🚨 **New Edge Case Logged**",
            f"• **Ground Truth:** {actual_label}",
            f"• **Ensemble AI Score:** {score_formatted}",
            f"• **Burstiness (CV):** {cv_formatted}",
            f"• **User Notes:** {note_text}",
            "• **Text Snippet:**",
            f"{clean_snippet}..."
        ]
        discord_msg = "\n".join(discord_lines)
        try:
            dc_res = requests.post(discord_url, json={"content": discord_msg})
            if dc_res.status_code in (200, 204):
                logged_anywhere = True
        except Exception as e:
            st.error(f"Discord logging failed: {e}")

    return logged_anywhere

# 4. Sidebar Controls
with st.sidebar:
    st.title("⚙️ Model Controls")
    st.markdown("**Ensemble Pipeline Active:**")
    st.caption("1. RoBERTa ChatGPT Detector (60%)")
    st.caption("2. OpenAI RoBERTa Detector (40%)")
    sensitivity = st.slider("Detection Sensitivity Threshold", 0.30, 0.95, 0.70, 0.05)
    st.markdown("---")
    st.markdown("**Guardrails Active:**")
    st.caption(f"• Word Limits: {MIN_WORD_COUNT} – {MAX_WORD_COUNT:,} words")
    st.caption(f"• Request Cooldown: {COOLDOWN_SECONDS} seconds")

# 5. Main UI Header & Inputs
st.title("Veridraft AI Detector Pro 🔍")
st.caption("Multi-model weighted ensemble & burstiness scoring pipeline")

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

analyze_btn = st.button("Run Ensemble Analysis", type="primary", use_container_width=True)

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
        
        with st.spinner("Executing multi-model ensemble & sentence highlight mapping..."):
            base_ai_prob, score_m1, score_m2 = ensemble_ai_predict(user_text)
            burstiness_cv, sentences = calculate_burstiness(user_text)

            calibrated_prob = base_ai_prob
            if burstiness_cv < 0.42:
                cv_delta = 0.42 - burstiness_cv
                calibrated_prob = min(0.992, max(0.94, base_ai_prob + (cv_delta * 4.0) + 0.80))

            highlighted_html = generate_sentence_highlights(sentences)

            st.session_state["analysis_result"] = {
                "text": user_text,
                "ai_prob": calibrated_prob,
                "score_m1": score_m1,
                "score_m2": score_m2,
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
    col1.metric("Ensemble AI Prob", f"{res['ai_prob']:.1%}")
    col2.metric("Burstiness (CV)", f"{res['burstiness_cv']:.3f}")
    col3.metric("Word Count", res['word_count'])
    col4.metric("Sentences", res['sentence_count'])

    if res['ai_prob'] >= sensitivity:
        st.error(f"🔴 **High Probability AI-Generated** (Exceeds {sensitivity:.0%} sensitivity threshold)")
    elif res['ai_prob'] >= 0.40:
        st.warning("🟡 **Mixed Origin / Likely AI-Assisted** (Contains structural edits or hybrid prose)")
    else:
        st.success("🟢 **High Probability Human-Written** (Natural sentence variance detected)")

    st.progress(res['ai_prob'])

    with st.expander("🔬 Model Ensemble Breakdown"):
        mb_col1, mb_col2 = st.columns(2)
        mb_col1.caption(f"**ChatGPT RoBERTa Detector:** {res['score_m1']:.1%}")
        mb_col2.caption(f"**OpenAI RoBERTa Detector:** {res['score_m2']:.1%}")

    st.markdown("### 🔍 Sentence-Level AI Map")
    st.caption("Hover over highlighted sentences to view individual AI confidence scores.")
    st.markdown(res["highlighted_html"], unsafe_allow_html=True)

    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.caption("🔴 **Red:** High AI Likelihood (≥65%)")
    col_l2.caption("🟡 **Yellow:** Moderate / Mixed Signals (35% - 64%)")
    col_l3.caption("🟢 **Green:** Likely Human-Written (<35%)")

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
                    st.success("Logged! Edge case saved to database and sent to Discord.")
