from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from detector import analyze_text, MAX_CHARS

app = FastAPI(title="Veridraft AI Detector API")

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze(payload: TextRequest):
    text = payload.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        ai_prob, human_prob, sentences, scores = analyze_text(text)
        return {
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "sentences": sentences,
            "scores": scores
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
