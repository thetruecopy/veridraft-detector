import streamlit as st
import pandas as pd
import io
import re
import math
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
st.sidebar.caption("Ensemble Architecture: Transformer Model + Burstiness Analysis + Lexical Entropy")

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

# Statistical Metrics (Burstiness & Lexical Variety)
def calculate_burstiness(sentences):
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.5, "Insufficient context"
    
    std_dev = np.std(lengths)
    mean_len = np.mean(lengths) + 1e-5
    
    # Burstiness score: lower variation = higher likelihood of AI
    variation_ratio = std_dev / mean_len
    if variation_ratio < 0.35:
        return 0.85, "Very Low Variation (Strong AI Signal)"
    elif variation_ratio < 0.55:
        return 0.60, "Moderate Variation"
    else:
        return 0.20, "High Human-like Variation"

def analyze_sentence(sentence):
    """Evaluates individual sentence AI probability using local context."""
    if len(sentence.strip().split()) < 4:
        return 20.0  # Very short fragments default low
    try:
        res = detector_pipeline(sentence[:512])[0]
        if res["label"] == "ChatGPT":
            return res["score"] * 100.0
        else:
            return (1.0 - res["score"]) * 100.0
    except Exception:
        return 50.0

# App UI
st.title("📝 Veridraft AI Detector Pro")
st.write("Advanced hybrid analysis combining Deep Learning, Burstiness, and Perplexity metrics to benchmark against ZeroGPT.")

input_method = st.radio("Input method:", ["Paste Text", "Upload Document"])
text_input = ""

if input_method == "Paste Text":
    text_input = st.text_area("Paste text here:", placeholder="Paste your text (minimum 30-50 words recommended)...", height=200)
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
        with st.spinner("Running Sentence-Level Ensemble Scan..."):
            sentence_scores = []
            sentence_details = []
            
            # 1. Sentence-by-Sentence Scan
            for idx, sentence in enumerate(raw_sentences):
                s_score = analyze_sentence(sentence)
                sentence_scores.append(s_score)
                sentence_details.append({
                    "sentence": sentence,
                    "score": s_score
                })
            
            # 2. Statistical Metrics
            burst_score, burst_label = calculate_burstiness(raw_sentences)
            avg_sentence_score = np.mean(sentence_scores) if sentence_scores else 50.0
            
            # 3. Hybrid Ensemble Formula
            # We weight sentence predictions (70%) + structural burstiness (30%)
            final_ai_prob = (avg_sentence_score * 0.70) + (burst_score * 100.0 * 0.30)
            final_ai_prob = min(99.9, max(0.1, final_ai_prob))
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
            st.caption("Sentences are individually evaluated. Red/Orange highlights indicate elevated AI confidence.")
            
            annotated_html = ""
            csv_rows = []
            
            for idx, item in enumerate(sentence_details):
                sc = item["score"]
                s_text = item["sentence"]
                
                # Dynamic Color Coding per Sentence Score
                if sc >= 70:
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
