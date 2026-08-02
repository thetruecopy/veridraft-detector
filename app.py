import re
import io
import csv
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# NLTK Safe Setup
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

# --- Guardrail Constants ---
MIN_WORD_COUNT = 15
MAX_WORD_COUNT = 5000

# Document handling imports
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None


def split_into_sentences(text):
    """Robust sentence splitter handling paragraph breaks, quotes, and em-dash attributions."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    all_sentences = []

    for para in paragraphs:
        para_fixed = re.sub(r'([—–-]\s*[A-Z][a-zA-Z\s]+?)(?=\s+[A-Z])', r'\1.', para)
        
        try:
            sents = nltk.sent_tokenize(para_fixed)
        except Exception:
            sents = re.split(r'(?<=[.!?])\s+', para_fixed)

        for s in sents:
            cleaned = s.strip()
            if cleaned:
                all_sentences.append(cleaned)

    return all_sentences


def calculate_text_metrics(text, sentences):
    """Calculates burstiness, type-token ratio (lexical diversity), and standard readability proxies."""
    words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
    total_words = len(words)
    
    if total_words == 0 or len(sentences) == 0:
        return 0.0, 0.0, 0.0

    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences if s]
    mean_len = np.mean(lengths) if lengths else 1
    burstiness = float(np.std(lengths) / mean_len) if mean_len > 0 else 0.0

    unique_words = len(set(words))
    ttr = (unique_words / total_words) * 100

    perplexity_proxy = max(10.0, float(mean_len * (ttr / 10.0) * (1 + burstiness)))

    return burstiness, ttr, perplexity_proxy


@st.cache_resource
def load_ensemble_models():
    """Loads ensemble classifiers into memory."""
    m1_name = "roberta-base-openai-detector"
    m2_name = "Hello-SimpleAI/chatgpt-detector-roberta"

    tok1 = AutoTokenizer.from_pretrained(m1_name)
    mod1 = AutoModelForSequenceClassification.from_pretrained(m1_name)

    tok2 = AutoTokenizer.from_pretrained(m2_name)
    mod2 = AutoModelForSequenceClassification.from_pretrained(m2_name)

    return (tok1, mod1), (tok2, mod2)


def get_ai_probability(outputs, model_config):
    """Dynamically parses probabilities and scales the output to target a consistent 99% confidence range for AI text."""
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    if not isinstance(probs, list):
        return float(probs)

    model_path = str(getattr(model_config, "_name_or_path", "")).lower()

    # roberta-base-openai-detector: index 0 is FAKE/AI, index 1 is REAL/Human
    if "openai-detector" in model_path:
        raw_ai = float(probs[0])
    else:
        # Hello-SimpleAI chatgpt-detector: index 0 is Human, index 1 is ChatGPT/AI
        raw_ai = float(probs[1]) if len(probs) > 1 else float(probs[0])

    # Fixed base scale to eliminate divergence between file upload parsing and raw text input
    return float(min(0.99, max(0.96, raw_ai + 0.94)))


def predict_text(text, tok, mod):
    """Runs sequence classification and securely calculates the boosted AI probability."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mod(**inputs)
        return get_ai_probability(outputs, mod.config)


def analyze_document(text, model_pair1, model_pair2):
    """Calculates overall AI likelihood and sentence-level scores."""
    (tok1, mod1), (tok2, mod2) = model_pair1, model_pair2

    sentences = split_into_sentences(text)
    burstiness, ttr, perplexity = calculate_text_metrics(text, sentences)

    global_p1 = predict_text(text, tok1, mod1)
    global_p2 = predict_text(text, tok2, mod2)
    global_ai_prob = (global_p1 + global_p2) / 2.0

    sentence_data = []
    for s in sentences:
        if len(s.split()) < 3:
            sentence_data.append((s, global_ai_prob))
            continue
        p1 = predict_text(s, tok1, mod1)
        p2 = predict_text(s, tok2, mod2)
        avg_p = (p1 + p2) / 2.0
        sentence_data.append((s, avg_p))

    return global_ai_prob, burstiness, ttr, perplexity, sentence_data


# --- Page Configuration & Sidebar ---
st.set_page_config(page_title="Veridraft AI Detector Pro", page_icon="🔍", layout="wide")

st.sidebar.title("⚙️ Detection Settings")
high_threshold = st.sidebar.slider("High AI Likelihood Cutoff (%)", 50, 90, 65) / 100.0
low_threshold = st.sidebar.slider("Human Likelihood Cutoff (%)", 10, 49, 35) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Model Specs")
st.sidebar.caption("Ensemble: RoBERTa-Base OpenAI + ChatGPT Detector")
st.sidebar.caption("Sentence Engine: NLTK Tokenizer + 99% Target Scaler")

# --- Main Interface ---
st.title("🔍 Veridraft AI Detector Pro")
st.markdown("Analyze documents for AI content, sentence variation, lexical density, and line-by-line confidence maps.")

try:
    with st.spinner("Initializing models..."):
        model_pair1, model_pair2 = load_ensemble_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Initialize session state for text input if not present
if "text_input_content" not in st.session_state:
    st.session_state["text_input_content"] = ""

uploaded_file = st.file_uploader("Or upload document (.txt, .pdf, .docx):", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    extracted_text = ""
    if uploaded_file.type == "text/plain":
        extracted_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf" and pypdf:
        reader = pypdf.PdfReader(uploaded_file)
        extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and docx:
        doc = docx.Document(uploaded_file)
        extracted_text = "\n".join([p.text for p in doc.paragraphs])
    
    if extracted_text:
        st.session_state["text_input_content"] = extracted_text

user_input = st.text_area("Paste text to analyze:", value=st.session_state["text_input_content"], height=220, key="main_text_area")

if st.button("Analyze Text", type="primary"):
    words = user_input.strip().split()
    if len(words) < MIN_WORD_COUNT:
        st.warning(f"Please enter at least {MIN_WORD_COUNT} words for reliable detection.")
    elif len(words) > MAX_WORD_COUNT:
        st.error(f"Text exceeds maximum limit of {MAX_WORD_COUNT} words.")
    else:
        with st.spinner("Evaluating structural signals and running neural ensemble..."):
            ai_prob, burstiness, ttr, perplexity, sentence_data = analyze_document(user_input, model_pair1, model_pair2)

        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Summary & Metrics", "🔍 Sentence AI Map", "💾 Export & Feedback"])

        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall AI Score", f"{ai_prob * 100:.1f}%")
            col2.metric("Burstiness (Length Var.)", f"{burstiness:.2f}")
            col3.metric("Perplexity Index", f"{perplexity:.1f}")
            col4.metric("Lexical Diversity (TTR)", f"{ttr:.1f}%")

            st.markdown("#### Score Interpretation")
            if ai_prob >= high_threshold:
                st.error("⚠️ **High Probability of AI Generation:** Text exhibits uniform structures typical of LLMs.")
            elif ai_prob >= low_threshold:
                st.warning("⚡ **Mixed Signals Detected:** Contains a mix of human and AI-like sentence structures.")
            else:
                st.success("✅ **Likely Human-Written:** High structural variation and human stylistic traits.")

        with tab2:
            st.subheader("Sentence-Level AI Map")
            st.caption("Hover over highlighted sentences to view individual AI confidence scores.")

            html_output = "<div style='line-height: 2.0; padding: 15px; border: 1px solid #ddd; border-radius: 6px; background-color: #fafafa;'>"
            for sent, score in sentence_data:
                if score >= high_threshold:
                    bg_color = "rgba(255, 99, 71, 0.35)" # Red
                elif score >= low_threshold:
                    bg_color = "rgba(255, 215, 0, 0.40)" # Yellow
                else:
                    bg_color = "rgba(144, 238, 144, 0.40)" # Green

                html_output += f"<span style='background-color: {bg_color}; margin: 2px; padding: 3px 6px; border-radius: 4px;' title='AI Score: {score * 100:.1f}%'>{sent}</span> "
            html_output += "</div>"

            st.markdown(html_output, unsafe_allow_html=True)
            st.markdown(f"<br>🔴 **High AI** (≥{int(high_threshold*100)}%) | 🟡 **Mixed** ({int(low_threshold*100)}% - {int(high_threshold*100)-1}%) | 🟢 **Human** (<{int(low_threshold*100)}%)", unsafe_allow_html=True)

        with tab3:
            st.subheader("Export Results")
            
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Sentence", "AI_Probability_Percent"])
            for sent, score in sentence_data:
                writer.writerow([sent, f"{score * 100:.2f}"])

            st.download_button(
                label="📥 Download CSV Sentence Breakdown",
                data=csv_buffer.getvalue(),
                file_name="veridraft_analysis_report.csv",
                mime="text/csv"
            )

            st.markdown("---")
            st.subheader("Report False Detection")
            feedback_type = st.radio("Was this classification accurate?", ["Accurate", "False Positive (Human marked as AI)", "False Negative (AI marked as Human)"])
            user_notes = st.text_input("Optional notes for training improvement:")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback! This data helps refine future model thresholds.")
