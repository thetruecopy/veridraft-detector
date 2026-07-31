import streamlit as st
import requests
from detector import extract_text_from_pdf, extract_text_from_docx, MAX_CHARS

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="Veridraft AI Detector", page_icon="📝")

st.markdown("## Veridraft AI Detector")
st.write(f"Paste your text or upload a document (.pdf, .docx, .txt) to analyze AI vs. Human probabilities. *(Max {MAX_CHARS:,} characters)*")

input_method = st.radio("Choose input method:", ["Paste Text", "Upload Document"])

text_input = ""

if input_method == "Paste Text":
    text_input = st.text_area(
        "Enter text to analyze:",
        value="",
        placeholder="Please enter your text...",
        height=150
    )
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
        
        if text_input.strip():
            st.info(f"Successfully extracted text from `{uploaded_file.name}`.")

if st.button("Analyze Text"):
    if not text_input.strip():
        st.warning("Please enter or upload some text to analyze.")
    else:
        try:
            with st.spinner("Calling backend API for analysis..."):
                response = requests.post(API_URL, json={"text": text_input})
                
                if response.status_code != 200:
                    error_detail = response.json().get("detail", "An error occurred.")
                    st.error(error_detail)
                else:
                    data = response.json()
                    overall_ai = data["ai_probability"]
                    overall_human = data["human_probability"]
                    sentences = data["sentences"]
                    scores = data["scores"]

                    st.success("Analysis complete!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Overall AI Probability",
                            value=f"{overall_ai:.1f}%",
                        )
                    with col2:
                        st.metric(
                            label="Overall Human Probability",
                            value=f"{overall_human:.1f}%",
                        )

                    report_content = f"--- Veridraft AI Detection Report ---\n"
                    report_content += f"Overall AI Probability: {overall_ai:.1f}%\n"
                    report_content += f"Overall Human Probability: {overall_human:.1f}%\n\n"
                    report_content += "--- Sentence Breakdown ---\n"
                    for sentence, score in zip(sentences, scores):
                        report_content += f"[AI Prob: {score:.1f}%] {sentence}\n"

                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=report_content,
                        file_name="veridraft_report.txt",
                        mime="text/plain"
                    )

                    st.markdown("---")
                    st.subheader("Sentence Breakdown")

                    for sentence, score in zip(sentences, scores):
                        st.markdown(f"**AI Prob: {score:.1f}%** — {sentence}")
                        
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Make sure FastAPI is running.")
