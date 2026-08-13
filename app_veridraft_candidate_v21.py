import re
import io
import csv
import math
import numpy as np
import streamlit as st
import torch
import pandas as pd
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# NLTK Safe Setup
import nltk
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
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
    """Normalizes whitespace, line breaks, and invisible PDF artifacts to guarantee parity with pasted text."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = [line for line in lines if line]
    return " ".join(cleaned_lines)


def split_into_sentences(text):
    """Robust sentence splitter handling paragraph breaks, quotes, and em-dash attributions."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    all_sentences = []

    for para in paragraphs:
        para_fixed = re.sub(r"([—–-]\s*[A-Z][a-zA-Z\s]+?)(?=\s+[A-Z])", r"\1.", para)

        try:
            sents = nltk.sent_tokenize(para_fixed)
        except Exception:
            sents = re.split(r"(?<=[.!?])\s+", para_fixed)

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
        "delving into",
        "it is important to note",
        "furthermore, it is",
        "in conclusion, as",
        "sheds light on",
        "testament to",
        "beacon of",
        "navigating the landscape",
        "in order to effectively",
    ]

    filler_count = sum(text_lower.count(phrase) for phrase in filler_phrases)

    evasion_penalty = 0.0
    if filler_count >= 2:
        evasion_penalty += 0.15 * min(filler_count, 4)

    if burstiness < 0.25 and ttr > 55.0:
        evasion_penalty += 0.20

    return float(min(0.35, evasion_penalty))


def calculate_text_metrics(text, sentences):
    """Calculates burstiness, type-token ratio (lexical diversity), and readability proxies."""
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    total_words = len(words)

    if total_words == 0 or len(sentences) == 0:
        return 0.0, 0.0, 0.0

    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences if s]
    mean_len = np.mean(lengths) if lengths else 1
    burstiness = float(np.std(lengths) / mean_len) if mean_len > 0 else 0.0

    unique_words = len(set(words))
    ttr = (unique_words / total_words) * 100

    perplexity_proxy = max(10.0, float(mean_len * (ttr / 10.0) * (1 + burstiness)))

    return burstiness, ttr, perplexity_proxy


@st.cache_resource
def load_ensemble_models():
    """Loads English AI ensemble classifiers into memory."""
    m1_name = "roberta-base-openai-detector"
    m2_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    m3_name = "fakespot-ai/roberta-base-ai-text-detection-v1"

    tok1 = AutoTokenizer.from_pretrained(m1_name)
    mod1 = AutoModelForSequenceClassification.from_pretrained(m1_name)

    tok2 = AutoTokenizer.from_pretrained(m2_name)
    mod2 = AutoModelForSequenceClassification.from_pretrained(m2_name)

    tok3 = AutoTokenizer.from_pretrained(m3_name)
    mod3 = AutoModelForSequenceClassification.from_pretrained(m3_name)

    return (tok1, mod1), (tok2, mod2), (tok3, mod3)


def get_ai_probability(outputs, model_config):
    """Dynamically parses model probabilities into an AI probability."""
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    if not isinstance(probs, list):
        return float(probs)

    model_path = str(getattr(model_config, "_name_or_path", "")).lower()

    if "openai-detector" in model_path:
        raw_ai = float(probs[0])
    else:
        raw_ai = float(probs[1]) if len(probs) > 1 else float(probs[0])

    return float(max(0.0, min(1.0, raw_ai)))


def predict_text(text, tok, mod):
    """Runs sequence classification and returns AI probability."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mod(**inputs)
        return get_ai_probability(outputs, mod.config)


def _document_level_scores(text, model_pairs):
    """Shared document-level scoring used by normal analysis and batch calibration."""
    (tok1, mod1), (tok2, mod2), (tok3, mod3) = model_pairs

    sentences = split_into_sentences(text)
    burstiness, ttr, perplexity = calculate_text_metrics(text, sentences)
    chunks = chunk_text_by_sentences(sentences, max_words_per_chunk=400)

    chunk_scores = []
    model1_scores = []
    model2_scores = []
    model3_scores = []

    for chunk in chunks:
        p1 = predict_text(chunk, tok1, mod1)
        p2 = predict_text(chunk, tok2, mod2)
        p3 = predict_text(chunk, tok3, mod3)

        model1_scores.append(p1)
        model2_scores.append(p2)
        model3_scores.append(p3)
        chunk_scores.append((p1 + p2) / 2.0)

    base_ai_prob = float(np.mean(chunk_scores)) if chunk_scores else 0.5
    model1_avg = float(np.mean(model1_scores)) if model1_scores else 0.5
    model2_avg = float(np.mean(model2_scores)) if model2_scores else 0.5
    model3_avg = float(np.mean(model3_scores)) if model3_scores else 0.5

    experimental_3model_avg = (model1_avg + model2_avg + model3_avg) / 3.0
    experimental_weighted_avg = (
        0.05 * model1_avg
        + 0.20 * model2_avg
        + 0.75 * model3_avg
    )

    evasion_adjustment = check_evasion_heuristics(text, ttr, burstiness)
    global_ai_prob = float(min(0.99, max(0.05, base_ai_prob + evasion_adjustment)))

    diagnostics = {
        "model1_avg": model1_avg,
        "model2_avg": model2_avg,
        "model3_avg": model3_avg,
        "experimental_3model_avg": experimental_3model_avg,
        "experimental_weighted_avg": experimental_weighted_avg,
        "base_ai_prob": base_ai_prob,
        "evasion_adjustment": evasion_adjustment,
        "final_ai_prob": global_ai_prob,
    }

    return (
        sentences,
        burstiness,
        ttr,
        perplexity,
        global_ai_prob,
        diagnostics,
    )


def analyze_document(text, model_pairs):
    """Calculates overall AI likelihood and sentence-level scores using ensemble chunked processing."""
    (tok1, mod1), (tok2, mod2), _ = model_pairs

    (
        sentences,
        burstiness,
        ttr,
        perplexity,
        global_ai_prob,
        diagnostics,
    ) = _document_level_scores(text, model_pairs)

    evasion_adjustment = diagnostics["evasion_adjustment"]

    sentence_data = []
    for s in sentences:
        if len(s.split()) < 3:
            sentence_data.append((s, global_ai_prob))
            continue

        p1 = predict_text(s, tok1, mod1)
        p2 = predict_text(s, tok2, mod2)
        avg_p = (p1 + p2) / 2.0
        sentence_data.append(
            (s, float(min(0.99, max(0.05, avg_p + evasion_adjustment))))
        )

    return global_ai_prob, burstiness, ttr, perplexity, sentence_data, diagnostics


def analyze_calibration_sample(text, model_pairs):
    """Fast document-level calibration path that skips sentence-by-sentence scoring."""
    (
        _sentences,
        burstiness,
        ttr,
        perplexity,
        global_ai_prob,
        diagnostics,
    ) = _document_level_scores(text, model_pairs)

    return global_ai_prob, burstiness, ttr, perplexity, diagnostics


def programmatic_analyze_text(text, model_pairs, high_threshold=0.65, low_threshold=0.35):
    """Programmatic API endpoint function for external integrations and automated pipelines."""
    cleaned_text = normalize_extracted_text(text)
    words = cleaned_text.strip().split()

    if len(words) < MIN_WORD_COUNT:
        return {"error": f"Text too short. Minimum required words: {MIN_WORD_COUNT}"}
    if len(words) > MAX_WORD_COUNT:
        return {"error": f"Text exceeds maximum limit of {MAX_WORD_COUNT} words."}

    ai_prob, burstiness, ttr, perplexity, sentence_data, diagnostics = analyze_document(
        cleaned_text, model_pairs
    )

    if ai_prob >= high_threshold:
        classification = "High Probability of AI Generation"
    elif ai_prob >= low_threshold:
        classification = "Mixed Signals Detected"
    else:
        classification = "Likely Human-Written"

    return {
        "overall_ai_score_percent": round(ai_prob * 100, 2),
        "classification": classification,
        "diagnostics": diagnostics,
        "metrics": {
            "burstiness": round(burstiness, 2),
            "perplexity": round(perplexity, 1),
            "lexical_diversity_ttr_percent": round(ttr, 2),
        },
        "sentence_count": len(sentence_data),
        "sentences": [
            {"sentence": s, "ai_score_percent": round(score * 100, 2)}
            for s, score in sentence_data
        ],
    }


def score_band(score, high_threshold, low_threshold):
    if score >= high_threshold:
        return "AI"
    if score >= low_threshold:
        return "Mixed"
    return "Human"


def candidate_decision_v1(diagnostics, burstiness):
    """
    Experimental rule derived from the separate 40-sample calibration set.

    IMPORTANT:
    - Diagnostic only.
    - Does NOT change production Final AI.
    - Uses model agreement/disagreement plus burstiness instead of one fixed weighted average.
    """
    model1 = float(diagnostics["model1_avg"])
    model2 = float(diagnostics["model2_avg"])
    fakespot = float(diagnostics["model3_avg"])
    burst = float(burstiness)

    # Strong AI path:
    # Model 1 is low, text has enough sentence-length variation,
    # and Fakespot is extremely high OR Model 2 strongly agrees with Fakespot.
    if (
        model1 < 0.15
        and burst >= 0.18
        and (
            (model2 >= 0.70 and fakespot >= 0.90)
            or fakespot >= 0.95
        )
    ):
        return "AI"

    # Strong Human path:
    # Model 2 is low and at least one human-supporting condition is present.
    if (
        model2 < 0.20
        and (
            model1 >= 0.15
            or burst < 0.18
            or fakespot < 0.20
        )
    ):
        return "Human"

    # Everything else remains deliberately uncertain.
    return "Mixed"



def candidate_v21_score(diagnostics, burstiness, ttr):
    """
    Frozen Candidate v2.1 meta-classifier.

    IMPORTANT:
    - Developer / diagnostic only.
    - Does NOT change production Final AI.
    - Coefficients are frozen from the final 6-feature Logistic Regression
      fitted on the separate 40-sample calibration set.
    """
    model1_ai = float(diagnostics["model1_avg"])
    model2_ai = float(diagnostics["model2_avg"])
    fakespot_ai = float(diagnostics["model3_avg"])
    heuristic_adjustment = float(diagnostics["evasion_adjustment"])
    burst = float(burstiness)
    lexical_ttr = float(ttr)

    logit = (
        -2.5423940522
        - 6.02256947 * model1_ai
        + 1.18992616 * model2_ai
        + 3.03814313 * fakespot_ai
        - 2.78631417 * heuristic_adjustment
        + 1.43381280 * burst
        + 0.00590290623 * lexical_ttr
    )

    ai_probability = 1.0 / (1.0 + math.exp(-logit))

    # Frozen Candidate v2.1 uncertainty bands:
    # Human < 50%, Mixed 50–64.99%, AI >= 65%
    if ai_probability >= 0.65:
        band = "AI"
    elif ai_probability >= 0.50:
        band = "Mixed"
    else:
        band = "Human"

    return float(ai_probability), band


def build_band_metrics(results_df, band_column):
    """Metrics for an already-classified AI/Mixed/Human band column."""
    actual = results_df["actual_label"].astype(str)
    predicted = results_df[band_column].astype(str)

    ai_mask = actual == "AI"
    human_mask = actual == "Human"

    ai_total = int(ai_mask.sum())
    human_total = int(human_mask.sum())

    ai_recall = (
        float(((predicted == "AI") & ai_mask).sum()) / ai_total if ai_total else 0.0
    )
    human_specificity = (
        float(((predicted == "Human") & human_mask).sum()) / human_total
        if human_total
        else 0.0
    )

    false_positives = int(((predicted == "AI") & human_mask).sum())
    false_negatives = int(((predicted == "Human") & ai_mask).sum())
    mixed_ai = int(((predicted == "Mixed") & ai_mask).sum())
    mixed_human = int(((predicted == "Mixed") & human_mask).sum())

    strict_correct = (
        ((actual == "AI") & (predicted == "AI"))
        | ((actual == "Human") & (predicted == "Human"))
    )
    strict_accuracy = float(strict_correct.mean()) if len(results_df) else 0.0

    confusion = pd.crosstab(actual, predicted, dropna=False).reindex(
        index=["AI", "Human"],
        columns=["AI", "Mixed", "Human"],
        fill_value=0,
    )

    return {
        "strict_accuracy": strict_accuracy,
        "ai_recall": ai_recall,
        "human_specificity": human_specificity,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "mixed_ai": mixed_ai,
        "mixed_human": mixed_human,
        "confusion": confusion,
    }


def build_calibration_metrics(results_df, score_column, high_threshold, low_threshold):
    """Builds strict three-band evaluation metrics for a chosen score column."""
    predicted = results_df[score_column].apply(
        lambda x: score_band(float(x), high_threshold, low_threshold)
    )
    actual = results_df["actual_label"]

    strict_correct = ((actual == "AI") & (predicted == "AI")) | (
        (actual == "Human") & (predicted == "Human")
    )

    ai_mask = actual == "AI"
    human_mask = actual == "Human"

    ai_total = int(ai_mask.sum())
    human_total = int(human_mask.sum())

    ai_recall = (
        float(((predicted == "AI") & ai_mask).sum()) / ai_total if ai_total else 0.0
    )
    human_specificity = (
        float(((predicted == "Human") & human_mask).sum()) / human_total
        if human_total
        else 0.0
    )

    false_positives = int(((predicted == "AI") & human_mask).sum())
    false_negatives = int(((predicted == "Human") & ai_mask).sum())
    mixed_ai = int(((predicted == "Mixed") & ai_mask).sum())
    mixed_human = int(((predicted == "Mixed") & human_mask).sum())

    strict_accuracy = float(strict_correct.mean()) if len(results_df) else 0.0

    confusion = pd.crosstab(actual, predicted, dropna=False).reindex(
        index=["AI", "Human"],
        columns=["AI", "Mixed", "Human"],
        fill_value=0,
    )

    return {
        "strict_accuracy": strict_accuracy,
        "ai_recall": ai_recall,
        "human_specificity": human_specificity,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "mixed_ai": mixed_ai,
        "mixed_human": mixed_human,
        "confusion": confusion,
    }


def render_calibration_lab(active_models, high_threshold, low_threshold):
    """Developer-only batch calibration workspace."""
    st.subheader("🧪 Calibration Lab")
    st.caption(
        "Batch-test labeled samples against VeriDraft diagnostics without changing production scoring."
    )

    calibration_file = st.file_uploader(
        "Upload calibration CSV",
        type=["csv"],
        key="calibration_csv",
    )

    st.markdown("**Required CSV columns:** `text`, `label`")
    st.caption("Use `AI` or `Human` in the label column. Optional column: `sample_id`.")

    if calibration_file is None:
        return

    try:
        calibration_df = pd.read_csv(calibration_file)
    except Exception as e:
        st.error(f"Could not read calibration CSV: {e}")
        return

    required_columns = {"text", "label"}
    if not required_columns.issubset(calibration_df.columns):
        st.error("CSV must contain both `text` and `label` columns.")
        return

    if calibration_df.empty:
        st.error("Calibration CSV contains no samples.")
        return

    calibration_df = calibration_df.copy()
    calibration_df["text"] = calibration_df["text"].astype(str)
    calibration_df["label_normalized"] = (
        calibration_df["label"].astype(str).str.strip().str.lower()
    )

    invalid_labels = calibration_df[
        ~calibration_df["label_normalized"].isin(["ai", "human"])
    ]
    if not invalid_labels.empty:
        st.error("Label column must contain only `AI` or `Human`.")
        st.dataframe(invalid_labels[["text", "label"]].head())
        return

    calibration_df["actual_label"] = calibration_df["label_normalized"].map(
        {"ai": "AI", "human": "Human"}
    )

    if "sample_id" not in calibration_df.columns:
        calibration_df["sample_id"] = [f"S{i + 1}" for i in range(len(calibration_df))]

    word_counts = calibration_df["text"].apply(lambda x: len(str(x).strip().split()))
    invalid_length_mask = (word_counts < MIN_WORD_COUNT) | (word_counts > MAX_WORD_COUNT)
    if invalid_length_mask.any():
        bad_rows = calibration_df.loc[
            invalid_length_mask, ["sample_id", "actual_label", "text"]
        ].copy()
        bad_rows["word_count"] = word_counts[invalid_length_mask].values
        st.error(
            f"{int(invalid_length_mask.sum())} sample(s) violate the {MIN_WORD_COUNT}-{MAX_WORD_COUNT} word limit."
        )
        st.dataframe(bad_rows.head(20))
        return

    st.success(f"Calibration file loaded successfully: {len(calibration_df)} samples.")
    st.dataframe(
        calibration_df[["sample_id", "actual_label", "text"]].head(10),
        use_container_width=True,
    )

    if st.button("Run Calibration", type="primary", key="run_calibration"):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for idx, row in calibration_df.reset_index(drop=True).iterrows():
            sample_id = str(row["sample_id"])
            actual_label = str(row["actual_label"])
            sample_text = normalize_extracted_text(str(row["text"]))

            status.write(f"Processing {idx + 1} of {len(calibration_df)}: {sample_id}")

            try:
                (
                    current_final,
                    burstiness,
                    ttr,
                    perplexity,
                    diagnostics,
                ) = analyze_calibration_sample(sample_text, active_models)

                weighted_score = float(diagnostics["experimental_weighted_avg"])
                current_band = score_band(current_final, high_threshold, low_threshold)
                weighted_band = score_band(weighted_score, high_threshold, low_threshold)
                candidate_band_v1 = candidate_decision_v1(diagnostics, burstiness)
                candidate_v21_probability, candidate_v21_band = candidate_v21_score(
                    diagnostics,
                    burstiness,
                    ttr,
                )

                results.append(
                    {
                        "sample_id": sample_id,
                        "actual_label": actual_label,
                        "word_count": len(sample_text.split()),
                        "model1_ai": float(diagnostics["model1_avg"]),
                        "model2_ai": float(diagnostics["model2_avg"]),
                        "fakespot_ai": float(diagnostics["model3_avg"]),
                        "three_model_test": float(
                            diagnostics["experimental_3model_avg"]
                        ),
                        "weighted_test": weighted_score,
                        "base_ensemble": float(diagnostics["base_ai_prob"]),
                        "heuristic_adjustment": float(
                            diagnostics["evasion_adjustment"]
                        ),
                        "current_final": float(current_final),
                        "burstiness": float(burstiness),
                        "ttr": float(ttr),
                        "perplexity": float(perplexity),
                        "current_band": current_band,
                        "weighted_band": weighted_band,
                        "candidate_v1_band": candidate_band_v1,
                        "candidate_v21_probability": candidate_v21_probability,
                        "candidate_v21_band": candidate_v21_band,
                        "current_correct_strict": (
                            (actual_label == "AI" and current_band == "AI")
                            or (actual_label == "Human" and current_band == "Human")
                        ),
                        "weighted_correct_strict": (
                            (actual_label == "AI" and weighted_band == "AI")
                            or (actual_label == "Human" and weighted_band == "Human")
                        ),
                        "candidate_v1_correct_strict": (
                            (actual_label == "AI" and candidate_band_v1 == "AI")
                            or (actual_label == "Human" and candidate_band_v1 == "Human")
                        ),
                        "candidate_v21_correct_strict": (
                            (actual_label == "AI" and candidate_v21_band == "AI")
                            or (actual_label == "Human" and candidate_v21_band == "Human")
                        ),
                        "text": sample_text,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "sample_id": sample_id,
                        "actual_label": actual_label,
                        "error": str(e),
                        "text": sample_text,
                    }
                )

            progress.progress((idx + 1) / len(calibration_df))

        status.empty()
        results_df = pd.DataFrame(results)
        st.session_state["calibration_results_df"] = results_df

    results_df = st.session_state.get("calibration_results_df")
    if results_df is None or results_df.empty:
        return

    st.markdown("### Calibration Results")

    error_rows = results_df[results_df.get("error", pd.Series(index=results_df.index, dtype=object)).notna()] if "error" in results_df.columns else pd.DataFrame()
    valid_results = results_df.dropna(subset=["current_final", "weighted_test"], how="any") if {"current_final", "weighted_test"}.issubset(results_df.columns) else pd.DataFrame()

    if not error_rows.empty:
        st.warning(f"{len(error_rows)} sample(s) failed during model execution.")
        st.dataframe(error_rows[["sample_id", "actual_label", "error"]], use_container_width=True)

    if valid_results.empty:
        st.error("No valid calibration results were produced.")
        return

    display_df = valid_results.copy()
    percent_columns = [
        "model1_ai",
        "model2_ai",
        "fakespot_ai",
        "three_model_test",
        "weighted_test",
        "base_ensemble",
        "heuristic_adjustment",
        "current_final",
    ]
    for col in percent_columns:
        display_df[col] = (display_df[col] * 100).round(1)

    display_columns = [
        "sample_id",
        "actual_label",
        "model1_ai",
        "model2_ai",
        "fakespot_ai",
        "three_model_test",
        "weighted_test",
        "base_ensemble",
        "heuristic_adjustment",
        "current_final",
        "burstiness",
        "ttr",
        "current_band",
        "weighted_band",
        "candidate_v1_band",
        "candidate_v21_probability",
        "candidate_v21_band",
        "current_correct_strict",
        "weighted_correct_strict",
        "candidate_v1_correct_strict",
        "candidate_v21_correct_strict",
    ]
    st.dataframe(display_df[display_columns], use_container_width=True)

    current_metrics = build_calibration_metrics(
        valid_results, "current_final", high_threshold, low_threshold
    )
    weighted_metrics = build_calibration_metrics(
        valid_results, "weighted_test", high_threshold, low_threshold
    )
    candidate_v1_metrics = build_band_metrics(
        valid_results, "candidate_v1_band"
    )
    candidate_v21_metrics = build_band_metrics(
        valid_results, "candidate_v21_band"
    )

    st.markdown("### Aggregate Metrics")
    metric_table = pd.DataFrame(
        [
            {
                "Scoring": "Current Final",
                "Strict Accuracy": current_metrics["strict_accuracy"],
                "AI Recall": current_metrics["ai_recall"],
                "Human Specificity": current_metrics["human_specificity"],
                "False Positives": current_metrics["false_positives"],
                "False Negatives": current_metrics["false_negatives"],
                "AI → Mixed": current_metrics["mixed_ai"],
                "Human → Mixed": current_metrics["mixed_human"],
            },
            {
                "Scoring": "Weighted Test",
                "Strict Accuracy": weighted_metrics["strict_accuracy"],
                "AI Recall": weighted_metrics["ai_recall"],
                "Human Specificity": weighted_metrics["human_specificity"],
                "False Positives": weighted_metrics["false_positives"],
                "False Negatives": weighted_metrics["false_negatives"],
                "AI → Mixed": weighted_metrics["mixed_ai"],
                "Human → Mixed": weighted_metrics["mixed_human"],
            },
            {
                "Scoring": "Candidate v1",
                "Strict Accuracy": candidate_v1_metrics["strict_accuracy"],
                "AI Recall": candidate_v1_metrics["ai_recall"],
                "Human Specificity": candidate_v1_metrics["human_specificity"],
                "False Positives": candidate_v1_metrics["false_positives"],
                "False Negatives": candidate_v1_metrics["false_negatives"],
                "AI → Mixed": candidate_v1_metrics["mixed_ai"],
                "Human → Mixed": candidate_v1_metrics["mixed_human"],
            },
            {
                "Scoring": "Candidate v2.1",
                "Strict Accuracy": candidate_v21_metrics["strict_accuracy"],
                "AI Recall": candidate_v21_metrics["ai_recall"],
                "Human Specificity": candidate_v21_metrics["human_specificity"],
                "False Positives": candidate_v21_metrics["false_positives"],
                "False Negatives": candidate_v21_metrics["false_negatives"],
                "AI → Mixed": candidate_v21_metrics["mixed_ai"],
                "Human → Mixed": candidate_v21_metrics["mixed_human"],
            },
        ]
    )

    metric_table["Strict Accuracy"] = (metric_table["Strict Accuracy"] * 100).round(1)
    metric_table["AI Recall"] = (metric_table["AI Recall"] * 100).round(1)
    metric_table["Human Specificity"] = (
        metric_table["Human Specificity"] * 100
    ).round(1)

    st.dataframe(metric_table, use_container_width=True)

    st.markdown("#### Current Final Confusion Matrix")
    st.dataframe(current_metrics["confusion"], use_container_width=True)

    st.markdown("#### Weighted Test Confusion Matrix")
    st.dataframe(weighted_metrics["confusion"], use_container_width=True)

    st.markdown("#### Candidate v1 Confusion Matrix")
    st.dataframe(candidate_v1_metrics["confusion"], use_container_width=True)

    st.markdown("#### Candidate v2.1 Confusion Matrix")
    st.dataframe(candidate_v21_metrics["confusion"], use_container_width=True)

    export_df = valid_results.copy()
    st.download_button(
        "📥 Download Calibration Results CSV",
        data=export_df.to_csv(index=False),
        file_name="veridraft_calibration_results.csv",
        mime="text/csv",
        key="download_calibration_results",
    )


# --- Page Configuration & Sidebar ---
st.set_page_config(page_title="Veridraft AI Detector Pro", page_icon="🔍", layout="wide")

# --- Initialize Supabase Client First ---
from supabase import create_client


@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_supabase()


# --- Microsoft Clarity Analytics ---
def inject_clarity(project_id: str):
    clarity_code = f"""
    <script type="text/javascript">
        (function(c,l,a,r,i,t,y){{
            c[a]=c[a]||function(){{(c[a].q=c[a].q||[]).push(arguments)}};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        }})(window, document, "clarity", "script", "{project_id}");
    </script>
    """
    components.html(clarity_code, height=0, width=0)


try:
    CLARITY_ID = st.secrets.get("CLARITY_PROJECT_ID", "")
    if CLARITY_ID:
        inject_clarity(CLARITY_ID)
except Exception:
    pass

st.sidebar.title("⚙️ Detection Settings")
high_threshold = st.sidebar.slider("High AI Likelihood Cutoff (%)", 50, 90, 65) / 100.0
low_threshold = st.sidebar.slider("Human Likelihood Cutoff (%)", 10, 49, 35) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Developer Access")
show_api_tab = st.sidebar.checkbox("Enable Developer / API View", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Model Specs")
st.sidebar.caption("Ensemble: RoBERTa OpenAI + ChatGPT Detector + Fakespot diagnostic")
st.sidebar.caption("Engine: Chunked Processing + Evasion Shield + Enterprise API")

st.sidebar.markdown("---")
st.sidebar.caption("VeriDraft v1.0.9 · Candidate v2.1 Diagnostic · Build 2026.08.13")

# --- Main Interface ---
st.title("🔍 Veridraft AI Detector Pro")
st.markdown(
    "Analyze documents for AI content, sentence variation, lexical density, and line-by-line confidence maps."
)

try:
    with st.spinner("Initializing AI detection models..."):
        model_pair1, model_pair2, model_pair3 = load_ensemble_models()
        active_models = (model_pair1, model_pair2, model_pair3)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Or upload document (.txt, .pdf, .docx):",
    type=["txt", "pdf", "docx"],
)
user_input = st.text_area(
    "Or paste text to analyze:",
    height=150,
    placeholder="Paste text here or upload a document above...",
)

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
                raw_extracted = "\n".join(
                    [page.extract_text() for page in reader.pages if page.extract_text()]
                )
            elif (
                uploaded_file.type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                and docx
            ):
                doc = docx.Document(io.BytesIO(file_bytes))
                raw_extracted = "\n".join([p.text for p in doc.paragraphs])

            target_text = normalize_extracted_text(raw_extracted)
        except Exception as ex:
            st.error(f"Error reading uploaded file: {ex}")
    else:
        target_text = normalize_extracted_text(user_input)

    words = target_text.strip().split()

    if len(words) < MIN_WORD_COUNT:
        st.warning(
            f"Please provide or upload a document containing at least {MIN_WORD_COUNT} words for reliable detection. "
            f"(Current words: {len(words)})"
        )
    elif len(words) > MAX_WORD_COUNT:
        st.error(f"Text exceeds maximum limit of {MAX_WORD_COUNT} words.")
    else:
        with st.spinner("Evaluating structural signals and running AI detection ensemble..."):
            (
                ai_prob,
                burstiness,
                ttr,
                perplexity,
                sentence_data,
                diagnostics,
            ) = analyze_document(target_text, active_models)

        st.session_state["ai_prob"] = ai_prob
        st.session_state["burstiness"] = burstiness
        st.session_state["ttr"] = ttr
        st.session_state["perplexity"] = perplexity
        st.session_state["sentence_data"] = sentence_data
        st.session_state["target_text"] = target_text
        st.session_state["diagnostics"] = diagnostics


# --- Results ---
if "ai_prob" in st.session_state:
    ai_prob = st.session_state["ai_prob"]
    burstiness = st.session_state["burstiness"]
    ttr = st.session_state["ttr"]
    perplexity = st.session_state["perplexity"]
    sentence_data = st.session_state["sentence_data"]
    target_text = st.session_state["target_text"]
    diagnostics = st.session_state["diagnostics"]

    st.markdown("---")

    if show_api_tab:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 Summary & Metrics",
                "🔍 Sentence AI Map",
                "💾 Export & Feedback",
                "🚀 Enterprise API",
                "🧪 Calibration Lab",
            ]
        )
    else:
        tab1, tab2, tab3 = st.tabs(
            [
                "📊 Summary & Metrics",
                "🔍 Sentence AI Map",
                "💾 Export & Feedback",
            ]
        )

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall AI Score", f"{ai_prob * 100:.1f}%")
        col2.metric("Burstiness (Length Var.)", f"{burstiness:.2f}")
        col3.metric("Perplexity Index", f"{perplexity:.1f}")
        col4.metric("Lexical Diversity (TTR)", f"{ttr:.1f}%")

        if show_api_tab:
            st.markdown("### 🔬 Developer Diagnostics")
            candidate_live_v1 = candidate_decision_v1(diagnostics, burstiness)
            candidate_live_v21_probability, candidate_live_v21_band = candidate_v21_score(
                diagnostics,
                burstiness,
                ttr,
            )
            d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = st.columns(10)

            d1.metric("Model 1 AI", f"{diagnostics['model1_avg'] * 100:.1f}%")
            d2.metric("Model 2 AI", f"{diagnostics['model2_avg'] * 100:.1f}%")
            d3.metric("Fakespot AI", f"{diagnostics['model3_avg'] * 100:.1f}%")
            d4.metric(
                "3-Model Test",
                f"{diagnostics['experimental_3model_avg'] * 100:.1f}%",
            )
            d5.metric(
                "Weighted Test",
                f"{diagnostics['experimental_weighted_avg'] * 100:.1f}%",
            )
            d6.metric("Base Ensemble", f"{diagnostics['base_ai_prob'] * 100:.1f}%")
            d7.metric(
                "Heuristic Adj.",
                f"{diagnostics['evasion_adjustment'] * 100:+.1f}%",
            )
            d8.metric("Final AI", f"{diagnostics['final_ai_prob'] * 100:.1f}%")
            d9.metric("Candidate v1", candidate_live_v1)
            d10.metric(
                "Candidate v2.1",
                f"{candidate_live_v21_probability * 100:.1f}% · {candidate_live_v21_band}",
            )

        st.markdown("#### Score Interpretation")
        if ai_prob >= high_threshold:
            st.error(
                "⚠️ **High Probability of AI Generation:** Text exhibits uniform structures typical of LLMs."
            )
        elif ai_prob >= low_threshold:
            st.warning(
                "⚡ **Mixed Signals Detected:** Contains a mix of human and AI-like sentence structures."
            )
        else:
            st.success(
                "✅ **Likely Human-Written:** High structural variation and human stylistic traits."
            )

    with tab2:
        st.subheader("Sentence-Level AI Map")
        st.caption("Hover over highlighted sentences to view individual AI confidence scores.")

        html_output = (
            "<div style='line-height: 2.0; padding: 15px; border: 1px solid #ddd; "
            "border-radius: 6px; background-color: #fafafa;'>"
        )

        for sent, score in sentence_data:
            if score >= high_threshold:
                bg_color = "rgba(255, 99, 71, 0.35)"
            elif score >= low_threshold:
                bg_color = "rgba(255, 215, 0, 0.40)"
            else:
                bg_color = "rgba(144, 238, 144, 0.40)"

            html_output += (
                f"<span style='background-color: {bg_color}; margin: 2px; padding: 3px 6px; "
                f"border-radius: 4px;' title='AI Score: {score * 100:.1f}%'>{sent}</span> "
            )

        html_output += "</div>"
        st.markdown(html_output, unsafe_allow_html=True)
        st.markdown(
            f"<br>🔴 **High AI** (≥{int(high_threshold * 100)}%) | "
            f"🟡 **Mixed** ({int(low_threshold * 100)}% - {int(high_threshold * 100) - 1}%) | "
            f"🟢 **Human** (<{int(low_threshold * 100)}%)",
            unsafe_allow_html=True,
        )

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
            mime="text/csv",
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
            "Optional notes for training improvement:",
            key="feedback_user_notes",
        )

        if st.button("Submit Feedback", key="submit_feedback"):
            try:
                (
                    supabase.table("edge_cases")
                    .insert(
                        {
                            "actual_label": feedback_type,
                            "predicted_score": float(ai_prob),
                            "burstiness_cv": float(burstiness),
                            "user_notes": user_notes,
                            "text_snippet": target_text[:500],
                        }
                    )
                    .execute()
                )

                st.success(
                    "Thank you for your feedback! Data successfully logged to Supabase."
                )
            except Exception as e:
                st.error(f"Failed to log data to Supabase: {e}")

    if show_api_tab:
        with tab4:
            st.subheader("Enterprise API Integration")
            st.markdown(
                "You can invoke Veridraft programmatically in your custom backend services "
                "(FastAPI/Flask/LMS integration) using the built-in `programmatic_analyze_text` function."
            )

            api_payload = programmatic_analyze_text(
                target_text,
                active_models,
                high_threshold,
                low_threshold,
            )

            st.markdown("#### Sample JSON API Response for Current Text:")
            st.json(api_payload)

            st.markdown("#### FastAPI Server Implementation Snippet:")
            st.code(
                """
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
                """,
                language="python",
            )

        with tab5:
            render_calibration_lab(active_models, high_threshold, low_threshold)
