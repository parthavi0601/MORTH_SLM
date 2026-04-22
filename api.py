import os
import sys
import tempfile
from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))

from pytorch_ner import NERPredictor
from pytorch_qa import QAPredictor, FormGenerator, extract_from_text_regex

app = FastAPI(title="Transport SLM API", description="API for MoRTH Road Accident Reporting Tool")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
ner_predictor = None
qa_predictor = None
form_generator = FormGenerator()

@app.on_event("startup")
def load_models():
    global ner_predictor, qa_predictor
    try:
        if os.path.exists("models/ner_bilstm_crf.pt"):
            ner_predictor = NERPredictor(model_dir="models")
            ner_predictor.load()   # must call .load() to populate vocab + model weights
            print("NER Model loaded.")
    except Exception as e:
        print(f"Failed to load NER model: {e}")
        ner_predictor = None

    try:
        if os.path.exists("models/qa_neural.pt"):
            qa_predictor = QAPredictor(model_dir="models")
            print("QA Model loaded.")
    except Exception as e:
        print(f"Failed to load QA model: {e}")

class TextRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    question: str
    top_k: int = 3

class PDFRequest(BaseModel):
    form_type: str  # "FAR" or "DAR"
    data: Dict[str, Any]

@app.get("/api/status")
def get_status():
    return {
        "ner_ready": ner_predictor is not None,
        "qa_ready": qa_predictor is not None
    }

@app.post("/api/extract")
def extract_details(req: TextRequest):
    text = req.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Regex extraction (always runs, no model needed)
    try:
        regex_data = extract_from_text_regex(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regex extraction error: {e}")

    # NER extraction (only if model is loaded; errors fall back gracefully)
    ner_data = {}
    entities = {}
    if ner_predictor is not None:
        try:
            entities = ner_predictor.extract_entities(text)
            ner_data  = ner_predictor.extract_to_far_format(text)
        except Exception as e:
            print(f"NER inference failed (falling back to regex only): {type(e).__name__}: {e}")

    return {
        "regex_data": regex_data,
        "ner_data":   ner_data,
        "entities":   entities,
    }

@app.post("/api/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF, then run NER + regex on it."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    try:
        from pdf_processor import extract_text_from_pdf, clean_text
    except ImportError:
        raise HTTPException(status_code=501, detail="pdf_processor module not available")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = extract_text_from_pdf(tmp_path)
        text = clean_text(result.get("full_text", ""))
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the PDF")

    regex_data = extract_from_text_regex(text)
    ner_data = {}
    entities = {}
    if ner_predictor:
        entities = ner_predictor.extract_entities(text)
        ner_data = ner_predictor.extract_to_far_format(text)

    return {
        "extracted_text": text,
        "char_count": len(text),
        "regex_data": regex_data,
        "ner_data": ner_data,
        "entities": entities,
    }

@app.post("/api/qa")
def ask_question(req: QARequest):
    if not qa_predictor:
        raise HTTPException(status_code=503, detail="QA Model is not loaded")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    results = qa_predictor.answer(req.question, top_k=req.top_k)
    return {"results": results}

@app.get("/api/form-template")
def get_form_template(type: str = "FAR"):
    """Return the blank FAR or DAR template for the frontend to render."""
    fg = FormGenerator()
    if type.upper() == "FAR":
        return fg.get_far_template()
    elif type.upper() == "DAR":
        return fg.get_dar_template()
    else:
        raise HTTPException(status_code=400, detail="type must be FAR or DAR")

@app.post("/api/generate-pdf")
def generate_pdf(req: PDFRequest):
    try:
        if req.form_type == "FAR":
            from pdf_generator import generate_far_pdf
            pdf_bytes = generate_far_pdf(req.data)
        elif req.form_type == "DAR":
            from pdf_generator import generate_dar_pdf
            pdf_bytes = generate_dar_pdf(req.data)
        else:
            raise HTTPException(status_code=400, detail="Invalid form_type")

        import base64
        return {"pdf_base64": base64.b64encode(pdf_bytes).decode('utf-8')}
    except ImportError:
        raise HTTPException(status_code=501, detail="pdf_generator module not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
