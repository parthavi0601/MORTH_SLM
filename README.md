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

<img width="1919" height="869" alt="image" src="https://github.com/user-attachments/assets/17c45067-769c-4ab7-846e-c81d4553197b" />

<img width="1917" height="875" alt="image" src="https://github.com/user-attachments/assets/a2fef57f-0917-484f-b0ac-d4e23a240e36" />

<img width="1917" height="865" alt="image" src="https://github.com/user-attachments/assets/ed3d8386-72a9-45ba-8ca0-b55826e6180d" />

<img width="1043" height="805" alt="image" src="https://github.com/user-attachments/assets/b79f257d-44bf-47e5-8b08-427bc5931390" />
# Transport SLM (AI Road Accident Reporting Tool)

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)

An AI-powered incident reporting and legal assistance platform developed for the **Ministry of Road Transport and Highways (MoRTH)**. This tool automates the extraction of accident details from unstructured text (such as news reports and FIRs) and streamlines the generation of official FAR/DAR reports for the Claims Tribunal.

---

## 🏗 System Architecture

```mermaid
graph TD
    subgraph "Frontend Interface (Streamlit)"
        UI[Government Reporting Dashboard]
        Input[Text / PDF Upload]
        Chat[Legal Q&A Interface]
    end

    subgraph "Custom PyTorch Neural Networks"
        NER[BiLSTM-CRF NER Model<br/>128k Vocab, ~538K Params]
        QA[Siamese BiLSTM QA Model<br/>Attention Pooling, ~1.6M Params]
    end

    subgraph "Processing & Generation Pipelines"
        PDF_Ext[PDF Parsing Engine]
        PDF_Gen[ReportLab PDF Generator<br/>FAR / DAR Forms]
        Regex[Heuristic/Regex Fallback]
    end

    subgraph "Knowledge Base"
        Corpus[(Transport Laws, IPC, MV Act)]
    end

    Input --> |Raw Text/PDF| PDF_Ext
    Input --> |Text| NER
    PDF_Ext --> |Extracted Text| NER
    
    NER --> |Casualties, Vehicles, Locations| UI
    NER -.-> |Fallback if unconfident| Regex
    Regex -.-> UI
    
    UI --> |Structured Data| PDF_Gen
    PDF_Gen --> |Official FAR/DAR PDF| UI

    Chat --> |Regulatory Query| QA
    Corpus --> |Passage Embeddings| QA
    QA --> |Cosine Similarity Retrieval| Chat
    
    classDef gov fill:#0b2244,stroke:#fff,stroke-width:1px,color:#fff;
    classDef model fill:#e63946,stroke:#fff,stroke-width:1px,color:#fff;
    classDef process fill:#f1faee,stroke:#1d3557,stroke-width:1px,color:#1d3557;
    classDef db fill:#a8dadc,stroke:#1d3557,stroke-width:1px,color:#1d3557;

    class UI,Input,Chat gov;
    class NER,QA model;
    class PDF_Ext,PDF_Gen,Regex process;
    class Corpus db;
```

## ✨ Key Features

- **Automated Data Extraction**: Extracts critical entities (victims, vehicle registration numbers, collision types, IPC sections) from raw unstructured text.
- **Custom Deep Learning Models**: Engineered two neural networks from scratch rather than relying on external APIs:
  1. **Named Entity Recognition (NER)**: A BiLSTM-CRF architecture trained on transportation domain sequences.
  2. **Semantic QA Retriever**: A Siamese BiLSTM network with self-attention for precise retrieval of legal transport regulations.
- **Official Form Generation**: Instantly maps extracted data into fully formatted, ready-to-submit PDF reports (First Accident Report - Form 1 & Detailed Accident Report - Form VII).
- **Interactive Dashboard**: Professional, government-styled UI built with Streamlit for investigating officers and RTO staff.

## 🛠 Tech Stack

- **Deep Learning**: PyTorch, NumPy
- **Frontend**: Streamlit, Custom CSS
- **Data Processing**: Pandas, Regular Expressions (Regex)
- **Document Handling**: ReportLab (PDF generation), pdfplumber (PDF extraction)

## 🚀 Quick Start (Local Setup)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd transport-slm
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run pytorch_app.py
   ```

## ☁️ Streamlit Cloud Deployment

This project is fully optimized for Streamlit Community Cloud.
1. Push the code to a GitHub repository.
2. Log in to [share.streamlit.io](https://share.streamlit.io/).
3. Click "New App", select this repository, and set the Main file path to `pytorch_app.py`.
4. *Note: `requirements.txt` is already configured to download the CPU-only version of PyTorch to ensure it comfortably fits within Streamlit Cloud's memory limits.*

## 📁 Repository Structure

- `pytorch_app.py` - Main Streamlit application and UI logic.
- `pytorch_ner.py` - Architecture and training loop for the BiLSTM-CRF NER model.
- `pytorch_qa.py` - Architecture and training loop for the Siamese BiLSTM QA model.
- `pdf_generator.py` - Logic to dynamically generate formatted FAR and DAR PDF reports using ReportLab.
- `pdf_processor.py` - Utilities for parsing and cleaning uploaded PDF documents.
- `requirements.txt` - Python dependencies, optimized for deployment.
- `models/` - Pre-trained model weights (`.pt`) and vocabularies (`.pkl`).

---
*Developed for the Ministry of Road Transport & Highways (MoRTH) accident reporting workflow.*

