import streamlit as st
from detector import analyze_text, extract_text_from_pdf, extract_text_from_docx, MAX_CHARS

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
            with st.spinner("Analyzing sentences with AI model..."):
                overall_ai, overall_human, sentences, scores = analyze_text(text_input)

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

            st.markdown("---")
            st.subheader("Sentence Breakdown")

            for sentence, score in zip(sentences, scores):
                st.markdown(f"**AI Prob: {score:.1f}%** — {sentence}")
                
        except ValueError as e:
            st.error(str(e))
