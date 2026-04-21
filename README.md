# 🚦 Road Accident Assistant — Surface Transport SLM

> AI-powered accident reporting tool for Indian road transport — built for police officers, RTO staff, and government clerks.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

This project is a **Small Language Model (SLM)** application designed for the **Ministry of Road Transport & Highways (MoRTH)**, India. It automates the tedious process of filling accident report forms (FAR/DAR) by extracting key information from accident news articles, FIRs, and press releases using AI.

### Key Features

| Feature | Description |
|---|---|
| 🤖 **NER Entity Extraction** | BiLSTM-CRF model extracts: locations, casualties, vehicles, causes, IPC sections, and more |
| ❓ **Q&A System** | Neural passage retriever + built-in knowledge base for transport law questions |
| 📋 **Auto-fill FAR/DAR Forms** | Pre-fills First Accident Report and Detailed Accident Report forms |
| 📄 **PDF Processing** | Reads accident reports from PDF files |
| 🎨 **Professional UI** | Clean, modern Streamlit interface designed for non-technical government staff |

---

## 🏗️ Project Structure

```
transport_slm_fixed/
│
├── pytorch_app.py          # 🖥️  Streamlit web application (main entry point)
├── pytorch_ner.py          # 🤖  BiLSTM-CRF NER model (Task 1)
├── pytorch_qa.py           # ❓  Neural QA retriever + knowledge base (Task 2)
├── pytorch_train.py        # 🏋️  Training pipeline for both models
├── train_qa_only.py        # 🔧  Train only the QA model
├── pdf_processor.py        # 📄  PDF text extraction utilities
├── requirements.txt        # 📦  Python dependencies
├── README.md               # 📖  This file
├── .gitignore              # 🚫  Git ignore rules
│
├── data/                   # 📊  Training data (not tracked in git)
│   ├── pdfs/               #     PDF documents for training
│   ├── *.csv               #     Accident news CSV files
│   └── *.xlsx              #     Structured crash data
│
└── models/                 # 🧠  Trained model weights (not tracked in git)
    ├── ner_bilstm_crf.pt   #     NER model weights
    ├── ner_vocab.pkl        #     NER vocabulary
    ├── ner_labels.pkl       #     NER label mappings
    ├── qa_neural.pt         #     QA model weights
    ├── qa_vocab.pkl         #     QA vocabulary
    └── qa_index.pkl         #     Pre-computed passage embeddings
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

Place your PDF documents (Motor Vehicles Act, MoRTH reports, etc.) in `data/pdfs/` and run:

```bash
python pytorch_train.py --pdf-dir data/pdfs --models-dir models
```

**Optional:** Add accident news CSV for improved NER accuracy:

```bash
python pytorch_train.py --pdf-dir data/pdfs --csv-file data/accidents.csv
```

### 3. Run the Application

```bash
python -m streamlit run pytorch_app.py
```

The app will open at `http://localhost:8501`

---

## 🧠 Model Architecture

### Task 1: Named Entity Recognition (NER)

| Component | Details |
|---|---|
| **Architecture** | Word Embedding → BiLSTM → CRF |
| **Parameters** | ~2.5M+ |
| **Entities** | `NUM_KILLED`, `NUM_INJURED`, `LOCATION`, `STATE`, `VEHICLE`, `COLLISION`, `CAUSE`, `ROAD_TYPE`, `DATE`, `TIME`, `WEATHER`, `IPC_SECTION`, `REG_NUMBER`, `SEVERITY` |
| **Training** | Auto-annotated from transport law PDFs + accident news CSVs |

### Task 2: Question Answering (QA)

| Component | Details |
|---|---|
| **Architecture** | Word Embedding → BiLSTM → Attention → Dual Encoder (Siamese) |
| **Parameters** | ~1.65M |
| **Retrieval** | Cosine similarity + BM25 keyword re-ranking |
| **Knowledge Base** | 12 built-in FAQs covering common transport law questions |
| **Answer Quality** | Sentence extraction + deduplication |

---

## 📋 Supported Forms

- **FAR (First Accident Report)** — Form 1: Filed within 48 hours by the investigating officer
- **DAR (Detailed Accident Report)** — Form VII: Filed within 90 days with complete investigation details

---

## 📊 Data Sources

The models can be trained on:
- **Motor Vehicles Act, 1988** (and 2019 Amendment)
- **National Highways Act & Rules**
- **MoRTH Annual Reports** on road accidents
- **IRC (Indian Roads Congress)** accident recording forms
- **Accident news datasets** (CSV format)
- **Structured crash data** (Excel format from state police)

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `--pdf-dir` | `data/pdfs` | Directory containing training PDFs |
| `--csv-file` | `None` | CSV file with accident news text |
| `--models-dir` | `models` | Directory to save trained models |
| `--ner-epochs` | `50` | NER training epochs |
| `--qa-epochs` | `30` | QA training epochs |
| `--max-samples` | `3000` | Max NER training samples per epoch |

---

## 🛡️ Privacy & Security

- ✅ **All data processed locally** — no information sent to external servers
- ✅ **No cloud APIs** — models run entirely on your machine
- ✅ **No user data stored** — uploaded PDFs are processed and deleted

---

## 📝 License

This project is for educational and government use under the Ministry of Road Transport & Highways (MoRTH), Government of India.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request
