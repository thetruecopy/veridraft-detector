import re
from transformers import pipeline
import streamlit as st
from pypdf import PdfReader
from docx import Document

@st.cache_resource
def load_detector():
    return pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text

def analyze_text(text_input):
    detector = load_detector()
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text_input)
        if s.strip()
    ]

    sentence_scores = []
    for sentence in sentences:
        result = detector(sentence)[0]
        label = result['label'].lower()
        score = result['score'] * 100

        if 'chatgpt' in label or 'ai' in label or 'fake' in label or label == 'label_1':
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

    return overall_ai_probability, overall_human_probability, sentences, sentence_scores
