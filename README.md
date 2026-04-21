# Road Accident Assistant — Surface Transport SLM

AI-powered accident reporting tool for Indian road transport.  
Built for police officers, RTO staff, and government clerks.

## What It Does

- **NER (Named Entity Recognition):** Extracts key details from accident news — location, casualties, vehicles, causes, IPC sections
- **Q&A System:** Answers questions about traffic rules and the Motor Vehicles Act
- **Auto-fill Forms:** Pre-fills FAR (First Accident Report) and DAR (Detailed Accident Report) forms
- **PDF Processing:** Reads accident reports from PDF files

## How To Run

### Install dependencies
```
pip install -r requirements.txt
```

### Train the models
Place your PDF documents in `data/pdfs/` and run:
```
python pytorch_train.py --pdf-dir data/pdfs --models-dir models
```

### Start the app
```
python -m streamlit run pytorch_app.py
```
App opens at http://localhost:8501

## Project Files

- `pytorch_app.py` — Streamlit web app (main entry point)
- `pytorch_ner.py` — BiLSTM-CRF NER model
- `pytorch_qa.py` — Neural QA retriever + built-in knowledge base
- `pytorch_train.py` — Training pipeline for both models
- `train_qa_only.py` — Train only the QA model
- `pdf_processor.py` — PDF text extraction
- `requirements.txt` — Python dependencies

## Models

| Model | Architecture | Parameters |
|-------|-------------|-----------|
| NER | BiLSTM-CRF | ~2.5M |
| QA | BiLSTM + Attention (Dual Encoder) | ~1.65M |

## Tech Stack

- Python 3.8+
- PyTorch
- Streamlit
- pdfplumber
