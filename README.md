# 📝 Veridraft AI Detector Pro

Veridraft AI Detector Pro is an advanced, multi-layered AI text detection engine built with Python, HuggingFace Transformers, and Streamlit. It combines deep learning model predictions with statistical NLP metrics (Context-Aware Windowing & Sentence Burstiness Analysis) to accurately identify AI-generated content across modern Large Language Models (LLMs) like ChatGPT, GPT-4o, Claude 3.5, and Gemini.

---

## ✨ Features

* **🤖 Calibrated Hybrid Detection Engine:** Combines RoBERTa transformer embeddings with contextual sliding windows to eliminate probability dilution.
* **📊 Sentence Variation (Burstiness):** Analyzes sentence length variance to detect unnaturally uniform structural patterns typical of AI text.
* **🔍 Sentence-by-Sentence Forensic Analysis:** Evaluates each sentence in context and provides interactive, color-coded risk highlights.
* **📄 Universal Document Parsing:** Supports direct text input as well as **PDF**, **DOCX**, and **TXT** file uploads with built-in text normalization to guarantee identical cross-format results.
* **📥 Forensic CSV Reports:** Export detailed sentence-level risk scores and classifications for further investigation.
* **📈 Real-time Session Analytics:** Track total documents scanned and running average risk scores in the sidebar.
* **💬 Feedback Loop:** Built-in accuracy rating interface for continuous evaluation and testing.

---

## 🛡️ How It Works (Ensemble Architecture)

Traditional single-model classifiers often struggle with modern LLMs due to model drift. Veridraft solves this using a 3-layer approach:

1. **Text Normalization:** Strips document formatting artifacts (line breaks, tabs, extra spaces) to ensure clean tokenization.
2. **Context-Aware Sliding Window:** Evaluates target sentences along with surrounding context to preserve semantic flow.
3. **Non-Linear Document Calibration:** Dynamically scales overall document risk when a majority of sentences trigger AI signals, preventing false negatives.

---

## 🚀 Quickstart & Local Setup

### Prerequisites
* Python 3.9+
* Git

### Installation

1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/veridraft-detector.git
   cd veridraft-detector

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run app.py

---

## 📦 Tech Stack

* **Frontend/UI:** Streamlit
* **ML / Transformer Model:** HuggingFace Transformers (Hello-SimpleAI/chatgpt-detector-roberta)
* **Data Handling & Analytics:** pandas, numpy
* **Document Parsers:** pypdf, python-docx

---

## 📜 License

Distributed under the MIT License.
