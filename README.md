# AI Document Risk & Compliance Classifier

A Streamlit prototype that classifies document text into Low, Medium, or High risk categories and flags uncertain predictions for human review.

## Features
- Upload PDF, DOCX, TXT, or paste text
- TF-IDF + Logistic Regression
- Confidence threshold and manual-review flag
- Indicative terms for reviewer assistance
- Dashboard, evaluation report, confusion matrix
- CSV and JSON export

## Important
The included dataset is synthetic and small. Evaluation describes only these demonstration examples and is not evidence of real-world accuracy or compliance. This is not legal advice or an authoritative compliance determination. Review all predictions. Do not upload confidential, personal, privileged, or regulated documents without authorization and an approved environment. Scanned PDFs may need OCR, which is not included.

## Run on Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```
If activation is blocked:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Streamlit Community Cloud
Push this project to GitHub, create a Streamlit app from the repository, select branch `main` and main file `app.py`, then deploy.
