# AI Resume Screening System

Streamlit app that ranks resumes against a job description using sentence embeddings, then displays extracted candidate details.

## Features

- Upload multiple resumes in PDF or image formats.
- Extract text using PDF parsing and OCR.
- Rank resumes by semantic similarity to a job description.
- Show extracted fields such as name, email, phone, education, and experience.

## Tech Stack

- Python 3.10
- Streamlit
- spaCy (`en_core_web_sm`)
- sentence-transformers (`all-MiniLM-L6-v2`)
- PyPDF2
- pytesseract + Tesseract OCR

## Prerequisites

1. Python 3.10+
2. Tesseract OCR installed on your machine:
   - Windows: install from the official project and add it to PATH.
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`

If Tesseract is installed but not detected on Windows, set the executable path in code (example):

```python
# Example only, uncomment and adjust the path if needed.
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Installation

```bash
git clone https://github.com/rushikesh369/AI-resume-screening-system.git
cd AI-resume-screening-system

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
streamlit run resume_ranker.py
```

## Notes

- Current bias/fairness output is a placeholder message and not a production fairness metric.
- For best OCR quality, use clear, high-resolution resume images.

## Repository

- GitHub: https://github.com/rushikesh369/AI-resume-screening-system
