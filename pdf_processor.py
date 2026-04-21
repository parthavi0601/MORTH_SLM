"""
PDF Text Extraction Module
Extracts and cleans text from transport domain PDFs (Acts, Forms, Reports)
"""

import pdfplumber
import os
import re
import json


def extract_text_from_pdf(pdf_path: str) -> dict:
    """Extract text from a PDF file, page by page."""
    result = {"filename": os.path.basename(pdf_path), "pages": [], "full_text": ""}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                result["pages"].append({"page_num": i + 1, "text": text})
            result["full_text"] = "\n\n".join(p["text"] for p in result["pages"])
    except Exception as e:
        result["error"] = str(e)
    return result


def clean_text(text: str) -> str:
    """Clean extracted PDF text for NLP processing."""
    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove header/footer artifacts
    text = re.sub(r'IRC:\d+-\d+', '', text)
    text = re.sub(r'Contd\.?\s*\.?\s*\d+', '', text)
    return text.strip()


def extract_sentences(text: str) -> list:
    """Split text into sentences for corpus building."""
    # Split on period, question mark, or newline followed by capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Filter short/junk sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


def build_corpus_from_folder(folder_path: str) -> dict:
    """Process all PDFs in a folder and build a text corpus."""
    corpus = {"documents": [], "all_sentences": [], "total_chars": 0}

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in {folder_path}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"  Processing: {pdf_file}...")
        result = extract_text_from_pdf(pdf_path)

        if "error" not in result:
            cleaned = clean_text(result["full_text"])
            sentences = extract_sentences(cleaned)
            corpus["documents"].append({
                "filename": pdf_file,
                "text": cleaned,
                "num_pages": len(result["pages"]),
                "num_sentences": len(sentences),
                "num_chars": len(cleaned)
            })
            corpus["all_sentences"].extend(sentences)
            corpus["total_chars"] += len(cleaned)
            print(f"    -> {len(sentences)} sentences, {len(cleaned):,} chars")
        else:
            print(f"    -> ERROR: {result['error']}")

    print(f"\nCorpus built: {len(corpus['documents'])} docs, "
          f"{len(corpus['all_sentences'])} sentences, "
          f"{corpus['total_chars']:,} total characters")
    return corpus


def extract_form_fields_from_pdf(pdf_path: str) -> list:
    """Extract structured form fields from FAR/DAR PDFs."""
    result = extract_text_from_pdf(pdf_path)
    text = result.get("full_text", "")

    fields = []
    # Pattern: numbered items like "1. Date of Accident" or "5. Offending Vehicle Details"
    pattern = r'(\d+)\.\s+([A-Z][^\n]{3,80})'
    matches = re.findall(pattern, text)
    for num, field_name in matches:
        field_name = field_name.strip().rstrip('.')
        # Check if it has sub-options (like Yes/No, or vehicle types)
        fields.append({"field_number": int(num), "field_name": field_name})

    return fields


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            result = extract_text_from_pdf(path)
            print(f"Extracted {len(result['pages'])} pages, "
                  f"{len(result['full_text'])} characters")
            print("\n--- First 500 chars ---")
            print(result["full_text"][:500])
        elif os.path.isdir(path):
            corpus = build_corpus_from_folder(path)
    else:
        print("Usage: python pdf_processor.py <pdf_file_or_folder>")
