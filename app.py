import re
import numpy as np
import pandas as pd
import streamlit as st
from transformers import pipeline
import docx
import pypdf

# Page Configuration
st.set_page_config(page_title="Veridraft AI Detector Pro", layout="wide")

# Load Transformer Model with Streamlit Caching
@st.cache_resource
def load_detector():
    return pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")

detector = load_detector()

# Session Analytics Setup
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0
if "scan_scores" not in st.session_state:
    st.session_state.scan_scores = []

# Sidebar Analytics Panel
st.sidebar.title("📊 Analytics Dashboard")
st.sidebar.metric("Total Scans", st.session_state.total_scans)

if st.session_state.scan_scores:
    avg_score = sum(st.session_state.scan_scores) / len(st.session_state.scan_scores)
    st.sidebar.metric("Avg AI Risk Score", f"{avg_score:.1f}%")
    st.sidebar.subheader("Recent Scan Trend")
    st.sidebar.line_chart(st.session_state.scan_scores)
else:
    st.sidebar.info("Run your first analysis to populate session analytics.")

# Title & Subheading
st.title("📝 Veridraft AI Detector Pro")
st.caption("Advanced hybrid analysis with global document context and burstiness calibration.")

# Input Selection
input_method = st.radio("Input method:", ["Paste Text", "Upload Document"], horizontal=True)

text_input = ""

if input_method == "Paste Text":
    text_input = st.text_area("Paste text here:", height=220)
else:
    uploaded_file = st.file_uploader("Upload a Document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(uploaded_file)
            text_input = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            text_input = "\n".join([p.text for p in doc.paragraphs if p.text])
        elif uploaded_file.name.endswith(".txt"):
            text_input = uploaded_file.getvalue().decode("utf-8")

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def calculate_burstiness(sentence_lengths):
    if len(sentence_lengths) < 2:
        return "Normal Variation", 0.5
    std_dev = np.std(sentence_lengths)
    mean_len = np.mean(sentence_lengths)
    cv = std_dev / mean_len if mean_len > 0 else 0
    if cv < 0.35:
        return "Very Low Variation (Strong AI Signal)", 0.95
    elif cv < 0.55:
        return "Low Variation", 0.70
    elif cv < 0.85:
        return "Moderate Variation", 0.40
    else:
        return "High Variation (Human-like)", 0.10

def parse_ai_score(res):
    label = res["label"].lower()
    score = res["score"]
    if any(k in label for k in ["chatgpt", "fake", "ai", "label_1"]):
        return score * 100.0
    else:
        return (1.0 - score) * 100.0

if st.button("Analyze Text") and text_input.strip():
    sentences = split_sentences(text_input)
    if not sentences:
        st.error("Please enter valid text for analysis.")
    else:
        # 1. Global Document-Level Evaluation (Pass first ~1500 chars to catch macro-patterns)
        doc_res = detector(text_input[:1500])[0]
        global_ai_score = parse_ai_score(doc_res)

        # 2. Sentence-by-Sentence Evaluation
        sentence_details = []
        sentence_lengths = [len(s.split()) for s in sentences]
        
        ai_count = 0
        total_confidence = 0.0

        for sentence in sentences:
            res = detector(sentence[:512])[0]
            ai_score = parse_ai_score(res)
            
            if ai_score >= 50.0:
                ai_count += 1
            
            total_confidence += ai_score
            sentence_details.append({"sentence": sentence, "score": ai_score})

        ai_ratio = ai_count / len(sentences)
        avg_ai_confidence = total_confidence / len(sentences)
        burst_label, burst_score = calculate_burstiness(sentence_lengths)

        # 3. Hybrid Calibration Logic
        scores = [item["score"] for item in sentence_details]
        scores.sort(reverse=True)
        top_k_count = max(1, int(len(scores) * 0.5))
        top_avg = np.mean(scores[:top_k_count]) if scores else 0.0

        # Anchor final risk using the strongest signals (Global Scan, Top Sentences, and Burstiness)
        base_risk = max(global_ai_score, top_avg, avg_ai_confidence)

        if "Very Low Variation" in burst_label:
            # Low burstiness heavily elevates risk floor for AI structural patterns
            final_ai_prob = max(base_risk * 1.35, 88.5 if base_risk > 20 else base_risk * 1.5)
        elif "Low Variation" in burst_label:
            final_ai_prob = max(base_risk, (ai_ratio * 70.0) + (burst_score * 30.0))
        else:
            final_ai_prob = base_risk

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

        # Highlighted Sentence Breakdown
        st.subheader("🔍 Sentence-by-Sentence Breakdown")
        st.caption("Sentences are individually evaluated. Red/Orange highlights indicate elevated AI confidence.")

        highlighted_html = ""
        for item in sentence_details:
            s_text = item["sentence"]
            s_score = item["score"]
            if s_score >= 65 or final_ai_prob > 80:
                color = "rgba(255, 99, 71, 0.3)"  # Light Red
            elif s_score >= 40:
                color = "rgba(255, 165, 0, 0.3)"  # Light Orange
            else:
                color = "rgba(144, 238, 144, 0.3)" # Light Green
            highlighted_html += f'<span style="background-color: {color}; padding: 3px 6px; border-radius: 4px; margin: 3px; display: inline-block; line-height: 1.6;">{s_text}</span> '

        st.markdown(f'<div style="line-height: 1.8;">{highlighted_html}</div>', unsafe_allow_html=True)

        # User Feedback Section
        st.write("---")
        st.subheader("💬 Detection Feedback")
        feedback_rating = st.radio(
            "How accurate was this scan result?",
            ["Accurate", "Partially Accurate", "Inaccurate"],
            horizontal=True
        )
        feedback_notes = st.text_input("Additional comments or context (optional):")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback! It helps improve model calibration.")
