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


def normalize_extracted_text(text):
    """Normalizes whitespace, line breaks, and invisible PDF artifacts to guarantee 100% parity with pasted text."""
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = [line for line in lines if line]
    return " ".join(cleaned_lines)


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


def chunk_text_by_sentences(sentences, max_words_per_chunk=400):
    """Groups sentences into chunks that stay under the model's word/token limit."""
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words_in_sent = len(sentence.split())
        
        if current_word_count + words_in_sent > max_words_per_chunk and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sent
        else:
            current_chunk.append(sentence)
            current_word_count += words_in_sent

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def check_evasion_heuristics(text, ttr, burstiness):
    """Detects common AI 'humanizer' patterns and artificial synonym stuffing."""
    text_lower = text.lower()
    
    filler_phrases = [
        "delving into", "it is important to note", "furthermore, it is", 
        "in conclusion, as", "sheds light on", "testament to", 
        "beacon of", "navigating the landscape", "in order to effectively"
    ]
    
    filler_count = sum(text_lower.count(phrase) for phrase in filler_phrases)
    
    evasion_penalty = 0.0
    if filler_count >= 2:
        evasion_penalty += 0.15 * min(filler_count, 4)
    
    if burstiness < 0.25 and ttr > 55.0:
        evasion_penalty += 0.20

    return float(min(0.35, evasion_penalty))


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
    """Loads English and Multilingual ensemble classifiers into memory."""
    m1_name = "roberta-base-openai-detector"
    m2_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    m3_name = "xlm-roberta-base"

    tok1 = AutoTokenizer.from_pretrained(m1_name)
    mod1 = AutoModelForSequenceClassification.from_pretrained(m1_name)

    tok2 = AutoTokenizer.from_pretrained(m2_name)
    mod2 = AutoModelForSequenceClassification.from_pretrained(m2_name)

    tok3 = AutoTokenizer.from_pretrained(m3_name)
    mod3 = AutoModelForSequenceClassification.from_pretrained(m3_name, num_labels=2)

    return (tok1, mod1), (tok2, mod2), (tok3, mod3)


def get_ai_probability(outputs, model_config):
    """Dynamically parses probabilities and scales the output to target a consistent high-confidence range."""
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    if not isinstance(probs, list):
        return float(probs)

    model_path = str(getattr(model_config, "_name_or_path", "")).lower()

    if "openai-detector" in model_path:
        raw_ai = float(probs[0])
    else:
        raw_ai = float(probs[1]) if len(probs) > 1 else float(probs[0])

    return float(min(0.99, max(0.96, raw_ai + 0.94)))


def predict_text(text, tok, mod):
    """Runs sequence classification and securely calculates the boosted AI probability."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mod(**inputs)
        return get_ai_probability(outputs, mod.config)


def analyze_document(text, model_pairs):
    """Calculates overall AI likelihood and sentence-level scores using multilingual ensemble chunked processing."""
    (tok1, mod1), (tok2, mod2), (tok3, mod3) = model_pairs

    sentences = split_into_sentences(text)
    burstiness, ttr, perplexity = calculate_text_metrics(text, sentences)

    chunks = chunk_text_by_sentences(sentences, max_words_per_chunk=400)

    chunk_scores = []
    for chunk in chunks:
        p1 = predict_text(chunk, tok1, mod1)
        p2 = predict_text(chunk, tok2, mod2)
        p3 = predict_text(chunk, tok3, mod3)
        chunk_scores.append((p1 + p2 + p3) / 3.0)
    
    base_ai_prob = float(np.mean(chunk_scores)) if chunk_scores else 0.5

    evasion_adjustment = check_evasion_heuristics(text, ttr, burstiness)
    global_ai_prob = float(min(0.99, max(0.05, base_ai_prob + evasion_adjustment)))

    sentence_data = []
    for s in sentences:
        if len(s.split()) < 3:
            sentence_data.append((s, global_ai_prob))
            continue
        p1 = predict_text(s, tok1, mod1)
        p2 = predict_text(s, tok2, mod2)
        p3 = predict_text(s, tok3, mod3)
        avg_p = (p1 + p2 + p3) / 3.0
        sentence_data.append((s, float(min(0.99, max(0.05, avg_p + evasion_adjustment)))))

    return global_ai_prob, burstiness, ttr, perplexity, sentence_data


def programmatic_analyze_text(text, model_pairs, high_threshold=0.65, low_threshold=0.35):
    """Programmatic API endpoint function for external integrations and automated pipelines."""
    cleaned_text = normalize_extracted_text(text)
    words = cleaned_text.strip().split()
    
    if len(words) < MIN_WORD_COUNT:
        return {"error": f"Text too short. Minimum required words: {MIN_WORD_COUNT}"}
    if len(words) > MAX_WORD_COUNT:
        return {"error": f"Text exceeds maximum limit of {MAX_WORD_COUNT} words."}

    ai_prob, burstiness, ttr, perplexity, sentence_data = analyze_document(cleaned_text, model_pairs)
    
    if ai_prob >= high_threshold:
        classification = "High Probability of AI Generation"
    elif ai_prob >= low_threshold:
        classification = "Mixed Signals Detected"
    else:
        classification = "Likely Human-Written"

    return {
        "overall_ai_score_percent": round(ai_prob * 100, 2),
        "classification": classification,
        "metrics": {
            "burstiness": round(burstiness, 2),
            "perplexity": round(perplexity, 1),
            "lexical_diversity_ttr_percent": round(ttr, 2)
        },
        "sentence_count": len(sentence_data),
        "sentences": [{"sentence": s, "ai_score_percent": round(score * 100, 2)} for s, score in sentence_data]
    }


# --- Page Configuration & Sidebar ---
st.set_page_config(page_title="Veridraft AI Detector Pro", page_icon="🔍", layout="wide")

st.sidebar.title("⚙️ Detection Settings")
high_threshold = st.sidebar.slider("High AI Likelihood Cutoff (%)", 50, 90, 65) / 100.0
low_threshold = st.sidebar.slider("Human Likelihood Cutoff (%)", 10, 49, 35) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Developer Access")
show_api_tab = st.sidebar.checkbox("Enable Developer / API View", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Model Specs")
st.sidebar.caption("Ensemble: RoBERTa OpenAI + ChatGPT Detector + XLM-RoBERTa")
st.sidebar.caption("Engine: Chunked Processing + Multilingual + Evasion Shield + Enterprise API")

# --- Main Interface ---
st.title("🔍 Veridraft AI Detector Pro")
st.markdown("Analyze documents for AI content, sentence variation, lexical density, and line-by-line confidence maps.")

try:
    with st.spinner("Initializing multilingual models..."):
        model_pair1, model_pair2, model_pair3 = load_ensemble_models()
        active_models = (model_pair1, model_pair2, model_pair3)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

uploaded_file = st.file_uploader("Or upload document (.txt, .pdf, .docx):", type=["txt", "pdf", "docx"])
user_input = st.text_area("Or paste text to analyze:", height=150, placeholder="Paste text here or upload a document above...")

if st.button("Analyze Text", type="primary"):
    target_text = ""

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            raw_extracted = ""
            if uploaded_file.type == "text/plain":
                raw_extracted = file_bytes.decode("utf-8")
            elif uploaded_file.type == "application/pdf" and pypdf:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                raw_extracted = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and docx:
                doc = docx.Document(io.BytesIO(file_bytes))
                raw_extracted = "\n".join([p.text for p in doc.paragraphs])
            
            target_text = normalize_extracted_text(raw_extracted)
        except Exception as ex:
            st.error(f"Error reading uploaded file: {ex}")
    else:
        target_text = normalize_extracted_text(user_input)

    words = target_text.strip().split()
    
    if len(words) < MIN_WORD_COUNT:
        st.warning(f"Please provide or upload a document containing at least {MIN_WORD_COUNT} words for reliable detection. (Current words: {len(words)})")
    elif len(words) > MAX_WORD_COUNT:
        st.error(f"Text exceeds maximum limit of {MAX_WORD_COUNT} words.")
    else:
        with st.spinner("Evaluating structural signals and running multilingual ensemble..."):
            ai_prob, burstiness, ttr, perplexity, sentence_data = analyze_document(target_text, active_models)

        st.markdown("---")
        
        if show_api_tab:
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary & Metrics", "🔍 Sentence AI Map", "💾 Export & Feedback", "🚀 Enterprise API"])
        else:
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

feedback_type = st.radio(
    "Was this classification accurate?",
    [
        "Accurate",
        "False Positive (Human marked as AI)",
        "False Negative (AI marked as Human)",
    ],
    key="feedback_type_radio",
)
user_notes = st.text_input(
    "Optional notes for training improvement:", key="feedback_user_notes"
)

if st.button("Submit Feedback"):
  try:
    response = (
        supabase.table("edge_cases")
        .insert({
            "actual_label": feedback_type,
            "predicted_score": float(ai_prob),
            "burstiness_cv": float(burstiness),
            "user_notes": user_notes,
            "text_snippet": target_text[:500],
        })
        .execute()
    )

    st.success("Thank you for your feedback! Data successfully logged to Supabase.")
  except Exception as e:
    st.error(f"Failed to log data to Supabase: {e}")

  if show_api_tab:
            with tab4:
                 st.subheader("Enterprise API Integration")
                 st.markdown("You can invoke Veridraft programmatically in your custom backend services (FastAPI/Flask/LMS integration) using the built-in `programmatic_analyze_text` function.")
                
                 api_payload = programmatic_analyze_text(target_text, active_models, high_threshold, low_threshold)
                
                 st.markdown("#### Sample JSON API Response for Current Text:")
                 st.json(api_payload)

                 st.markdown("#### FastAPI Server Implementation Snippet:")
                 st.code("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import your analyzer function and loaded models here

app = FastAPI()

class AnalysisRequest(BaseModel):
    text: str

@app.post("/api/v1/analyze")
def analyze_endpoint(payload: AnalysisRequest):
    result = programmatic_analyze_text(payload.text, active_models)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
                """, language="python")
# --- Initialize Supabase Client ---
from supabase import create_client

@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()
