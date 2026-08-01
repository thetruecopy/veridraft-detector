import streamlit as st
import pandas as pd
import io
import re
import pypdf
import docx
from transformers import pipeline

# Set Page Configuration
st.set_page_config(page_title="Veridraft AI Detector", page_icon="📝", layout="wide")

# Initialize Session Analytics State
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0
if "scan_scores" not in st.session_state:
    st.session_state.scan_scores = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# Sidebar Analytics Dashboard
st.sidebar.title("📊 Session Analytics")
st.sidebar.metric("Total Documents Scanned", st.session_state.total_scans)

if st.session_state.scan_scores:
    avg_ai = sum(st.session_state.scan_scores) / len(st.session_state.scan_scores)
    st.sidebar.metric("Avg. AI Risk Score", f"{avg_ai:.1f}%")
else:
    st.sidebar.metric("Avg. AI Risk Score", "N/A")

# Cache Model Loader for fast inference on Streamlit Cloud
@st.cache_resource
def load_detector():
    return pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")

detector_pipeline = load_detector()

# Document Parsing Functions
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

# App Header
st.title("📝 Veridraft AI Detector")
st.write("Analyze text or uploaded documents (.pdf, .docx, .txt) to evaluate AI vs. Human writing probabilities.")

input_method = st.radio("Choose input method:", ["Paste Text", "Upload Document"])
text_input = ""

if input_method == "Paste Text":
    text_input = st.text_area("Paste text here:", placeholder="Please enter your text...", height=200)
else:
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            text_input = extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            text_input = extract_text_from_docx(uploaded_file)
        elif file_extension == "txt":
            text_input = uploaded_file.getvalue().decode("utf-8")

if st.button("Analyze Text", type="primary"):
    if not text_input.strip():
        st.warning("Please provide some text to analyze.")
    else:
        with st.spinner("Analyzing text with AI model..."):
            try:
                # Run Inference Directly
                truncated_text = text_input[:1500]
                results = detector_pipeline(truncated_text)
                label = results[0]["label"]
                score = results[0]["score"]
                
                if label == "ChatGPT":
                    ai_prob = score * 100
                else:
                    ai_prob = (1.0 - score) * 100
                    
                human_prob = 100.0 - ai_prob
                
                # Update Session Analytics
                st.session_state.total_scans += 1
                st.session_state.scan_scores.append(ai_prob)
                
                # Display Score Metrics
                col1, col2 = st.columns(2)
                col1.metric("🤖 AI Generated", f"{ai_prob:.1f}%")
                col2.metric("👤 Human Written", f"{human_prob:.1f}%")
                
                st.progress(ai_prob / 100.0)
                
                # Sentence-Level Analysis
                st.subheader("🔍 Sentence Analysis")
                sentences = re.split(r'(?<=[.!?])\s+', text_input)
                
                annotated_html = ""
                sentence_data = []
                
                for idx, s in enumerate(sentences):
                    if not s.strip():
                        continue
                    s_color = "rgba(255, 99, 71, 0.3)" if ai_prob > 50 else "rgba(144, 238, 144, 0.4)"
                    annotated_html += f'<span style="background-color: {s_color}; padding: 2px 4px; margin: 2px; border-radius: 4px; display: inline-block;">{s}</span> '
                    
                    sentence_data.append({
                        "Sentence #": idx + 1,
                        "Sentence Text": s,
                        "Estimated Flag": "AI Likely" if ai_prob > 50 else "Human Likely"
                    })
                
                st.markdown(annotated_html, unsafe_allow_html=True)
                
                # Export Analysis CSV
                st.write("---")
                df = pd.DataFrame(sentence_data)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Analysis Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="veridraft_analysis_report.csv",
                    mime="text/csv"
                )
                
                # Feedback Section
                st.write("---")
                st.subheader("💬 Feedback & Accuracy")
                feedback_rating = st.radio(
                    "Was this detection result accurate?",
                    ["Select an option", "Accurate", "False Positive (Marked Human text as AI)", "False Negative (Marked AI text as Human)"],
                    horizontal=True
                )
                
                feedback_notes = st.text_input("Additional comments (optional):")
                if st.button("Submit Feedback"):
                    if feedback_rating != "Select an option":
                        st.session_state.feedback_log.append({
                            "rating": feedback_rating,
                            "notes": feedback_notes
                        })
                        st.success("Thank you for your feedback! It helps improve model accuracy.")
                    else:
                        st.warning("Please select a rating option before submitting.")

            except Exception as e:
                st.error(f"Error during detection: {e}")
