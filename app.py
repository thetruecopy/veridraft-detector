import streamlit as st
import re
from transformers import pipeline

st.set_page_config(page_title="Veridraft AI Detector", page_icon="📝")

st.markdown("## Veridraft AI Detector")
st.write("Debugging raw model outputs and sentence-level predictions.")

@st.cache_resource
def load_detector():
    return pipeline("text-classification", model="ahmediqbal/ai-text-detector-model")

with st.spinner("Loading AI detection model..."):
    detector = load_detector()

text_input = st.text_area(
    "Enter text to analyze:",
    value="",
    placeholder="Please enter your text...",
    height=150
)

if st.button("Analyze Text"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentences with AI model..."):
            sentences = [
                s.strip()
                for s in re.split(r'(?<=[.!?])\s+', text_input)
                if s.strip()
            ]

            sentence_scores = []

            for sentence in sentences:
                result = detector(sentence)[0]
                label = result['label']
                score = result['score'] * 100

                # Display raw label for debugging
                st.write(f"**Raw Model Output:** Label: `{label}`, Score: `{score:.2f}%` for sentence: *{sentence}*")

                # Map based on standard binary classifier conventions
                if 'ai' in label.lower() or 'fake' in label.lower() or label == 'LABEL_1':
                    ai_prob = score
                else:
                    ai_prob = 100.0 - score

                sentence_scores.append(ai_prob)

            if sentence_scores:
                overall_ai_probability = sum(sentence_scores) / len(sentence_scores)
                overall_human_probability = 100.0 - overall_ai_probability
            else:
                overall_ai_probability = 0.0
                overall_human_probability = 100.0

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Overall AI Probability",
                    value=f"{overall_ai_probability:.1f}%",
                )
            with col2:
                st.metric(
                    label="Overall Human Probability",
                    value=f"{overall_human_probability:.1f}%",
                )
