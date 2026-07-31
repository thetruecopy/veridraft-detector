import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

st.set_page_config(page_title="VeriDraft AI Detector", page_icon="📝", layout="centered")

@st.cache_resource
def load_model():
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("📝 VeriDraft AI Detector")
st.markdown("Paste your text below to analyze sentence-level AI vs. Human probabilities.")

text = st.text_area(
    "Enter text to analyze:",
    "I love mangoes. Mangoes love me. I went to the market to buy yummy veggies, fruits, and groceries for the party. Let's have a party!"
)

if st.button("Analyze Text"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
            if not sentences:
                sentences = [text]

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
            
            ai_prob_overall = probs[0][1].item() * 100 if model.config.num_labels > 1 else probs[0][0].item() * 100
            human_prob_overall = 100 - ai_prob_overall

            st.success("Analysis complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall AI Probability", f"{ai_prob_overall:.1f}%")
            with col2:
                st.metric("Overall Human Probability", f"{human_prob_overall:.1f}%")

            st.markdown("---")
            st.subheader("Sentence Breakdown")

            for sent in sentences:
                sent_inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    sent_outputs = model(**sent_inputs)
                    sent_probs = F.softmax(sent_outputs.logits, dim=-1)
                
                sent_ai = sent_probs[0][1].item() * 100 if model.config.num_labels > 1 else sent_probs[0][0].item() * 100
                color = "red" if sent_ai > 50 else "green"
                st.markdown(f"<span style='color:{color}'>**AI Prob: {sent_ai:.1f}%**</span> — {sent}", unsafe_allow_html=True)
