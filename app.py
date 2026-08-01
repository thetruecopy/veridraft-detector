import streamlit as st
import pandas as pd
import io
import re
import numpy as np
import pypdf
import docx
from transformers import pipeline

# Set Page Configuration
st.set_page_config(page_title="Veridraft AI Detector Pro", page_icon="📝", layout="wide")

# Initialize Session Analytics
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0
if "scan_scores" not in st.session_state:
    st.session_state.scan_scores = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# Sidebar Analytics
st.sidebar.title("📊 Session Analytics")
st.sidebar.metric("Total Scans", st.session_state.total_scans)

if st.session_state.scan_scores:
    avg_ai = sum(st.session_state.scan_scores) / len(st.session_state.scan_scores)
    st.sidebar.metric("Avg. AI Risk Score", f"{avg_ai:.1f}%")
else:
    st.sidebar.metric("Avg. AI Risk Score", "N/A")

st.sidebar.write("---")
st.sidebar.subheader("🛡️ Detection Engine")
st.sidebar.caption("Ensemble v2: Context-Aware Windowing + Non-Linear Calibration")

# Cache Model Loader
@st.cache_resource
def load_detector():
    return pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")

detector_pipeline = load_detector()

# Document Parsers
def extract_text_from_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Statistical Metrics (Burstiness)
def calculate_burstiness(sentences):
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.5, "Insufficient Context"
    
    std_dev = np.std(lengths)
    mean_len = np.mean(lengths) + 1e-5
    variation_ratio = std_dev / mean_len
    
    if variation_ratio < 0.35:
        return 0.85, "Very Low Variation (Strong AI Signal)"
    elif variation_ratio < 0.55:
        return 0.50, "Moderate Variation"
    else:
        return 0.15, "High Human Variation"

def predict_ai_prob(text_chunk):
    """Helper to run model on text chunk safely."""
    if len(text_chunk.strip().split()) < 3:
        return 20.0
    try:
        res = detector_pipeline(text_chunk[:512])[0]
        if res["label"] == "ChatGPT":
            return res["score"] * 100.0
        else:
            return (1.0 - res["score"]) * 100.0
    except Exception:
        return 50.0

# App UI
st.title("📝 Veridraft AI Detector Pro")
st.write("Advanced hybrid analysis with context-aware windowing and calibrated document-level risk scoring.")

input_method = st.radio("Input method:", ["Paste Text", "Upload Document"])
text_input = ""

if input_method == "Paste Text":
    text_input = st.text_area("Paste text here:", placeholder="Paste your text (minimum 20-30 words recommended)...", height=200)
else:
    uploaded_file = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "pdf":
            text_input = extract_text_from_pdf(uploaded_file)
        elif file_ext == "docx":
            text_input = extract_text_from_docx(uploaded_file)
        elif file_ext == "txt":
            text_input = uploaded_file.getvalue().decode("utf-8")

if st.button("Analyze Text", type="primary"):
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_input) if s.strip()]
    word_count = len(text_input.split())
    
    if word_count < 15:
        st.warning("Please provide at least 15–20 words for an accurate analysis.")
    else:
        with st.spinner("Running Calibrated Context-Aware Scan..."):
            sentence_details = []
            ai_sentence_count = 0
            
            # 1. Context-Aware Sentence Scan (Sliding Window Context)
            for idx, sentence in enumerate(raw_sentences):
                # Build context window (sentence + previous + next sentence)
                prev_s = raw_sentences[idx - 1] if idx > 0 else ""
                next_s = raw_sentences[idx + 1] if idx < len(raw_sentences) - 1 else ""
                context_chunk = f"{prev_s} {sentence} {next_s}".strip()
                
                # Combine sentence prediction + context prediction
                s_score = predict_ai_prob(sentence)
                c_score = predict_ai_prob(context_chunk)
                
                # Weighted sentence score
                combined_score = (s_score * 0.4) + (c_score * 0.6)
                
                if combined_score >= 50.0:
                    ai_sentence_count += 1
                
                sentence_details.append({
                    "sentence": sentence,
                    "score": combined_score
                })
            
            # 2. Document Level Ratio Calibration (ZeroGPT style)
            total_sentences = len(raw_sentences)
            ai_ratio = ai_sentence_count / total_sentences
            avg_ai_confidence = np.mean([item["score"] for item in sentence_details if item["score"] >= 50.0]) if ai_sentence_count > 0 else 0
            
            burst_score, burst_label = calculate_burstiness(raw_sentences)
            
            # Non-linear Document Calibration Score:
            # If majority of sentences are AI (>=70%), document risk scales heavily towards 90%+
            if ai_ratio >= 0.70:
                base_risk = max(ai_ratio * 100.0, avg_ai_confidence)
                final_ai_prob = base_risk + (100.0 - base_risk) * 0.45  # Calibration boost
            elif ai_ratio >= 0.40:
                final_ai_prob = (ai_ratio * 70.0) + (burst_score * 30.0)
            else:
                final_ai_prob = np.mean([item["score"] for item in sentence_details])
                
            final_ai_prob = min(99.4, max(0.5, final_ai_prob))
            final_human_prob = 100.0 - final_ai_prob
            
            # Update Session Analytics
            st.session_state.total_scans += 1
            st.session_state.scan_scores.append(final_ai_prob)
            
            # Display Primary Results
            st.write("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("🤖 Overall AI Risk", f"{final_ai_prob:.1f}%")
            col2.metric("👤 Human Likelihood", f"{final_human_prob:.1f}%")
            col3.metric("📊 Sentence Variation (Burstiness)", burst_label)
            
            st.progress(final_ai_prob / 100.0)
            
            # 4. Highlighted Sentence Breakdown
            st.subheader("🔍 Sentence-by-Sentence Breakdown")
            st.caption("Sentences are individually evaluated with contextual windowing. Red/Orange highlights indicate elevated AI confidence.")
            
            annotated_html = ""
            csv_rows = []
            
            for idx, item in enumerate(sentence_details):
                sc = item["score"]
                s_text = item["sentence"]
                
                if sc >= 65:
                    bg_color = "rgba(255, 77, 77, 0.4)"   # High Risk Red
                    status = "AI Generated"
                elif sc >= 45:
                    bg_color = "rgba(255, 193, 7, 0.4)"   # Medium Risk Yellow/Orange
                    status = "Mixed / Unclear"
                else:
                    bg_color = "rgba(144, 238, 144, 0.4)" # Low Risk Green
                    status = "Human Written"
                
                annotated_html += f'<span style="background-color: {bg_color}; padding: 3px 6px; margin: 3px; border-radius: 4px; display: inline-block;" title="AI Confidence: {sc:.1f}%">{s_text}</span> '
                
                csv_rows.append({
                    "Sentence #": idx + 1,
                    "Sentence Text": s_text,
                    "AI Probability (%)": round(sc, 1),
                    "Classification": status
                })
            
            st.markdown(annotated_html, unsafe_allow_html=True)
            
            # CSV Download
            st.write("---")
            df = pd.DataFrame(csv_rows)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Detailed Forensic Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name="veridraft_forensic_report.csv",
                mime="text/csv"
            )
            
            # Feedback
            st.subheader("💬 Model Feedback Loop")
            feedback_rating = st.radio(
                "Rate accuracy for this scan:",
                ["Select option", "Accurate", "False Positive", "False Negative"],
                horizontal=True
            )
            if st.button("Submit Feedback"):
                if feedback_rating != "Select option":
                    st.session_state.feedback_log.append({"rating": feedback_rating})
                    st.success("Feedback logged! Thank you for refining our model.")
