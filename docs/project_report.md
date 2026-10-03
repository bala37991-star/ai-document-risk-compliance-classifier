# AI Document Risk & Compliance Classifier

## Abstract
This prototype extracts text from business documents and classifies it as Low, Medium, or High risk using TF-IDF features and Logistic Regression. Low-confidence predictions are flagged for manual review. A Streamlit dashboard supports review and export.

## Problem Statement
Manual review of large document collections can be time-consuming. A classification aid may help organize documents for review, but outputs must be checked by people.

## Objectives
- Extract text from common document formats.
- Classify into Low, Medium, and High risk.
- Flag uncertain predictions.
- Show indicative terms and model evaluation.
- Provide a dashboard and CSV/JSON exports.

## Methodology
1. Prepare labeled synthetic examples.
2. Extract and normalize text from PDF, DOCX, TXT, or pasted text.
3. Convert text to TF-IDF unigram and bigram features.
4. Train class-balanced Logistic Regression.
5. Predict class and maximum class score.
6. Flag scores below 0.60 for manual review.
7. Display results and export structured data.

## Technology Stack
Python, Streamlit, Pandas, scikit-learn, TF-IDF, Logistic Regression, pypdf, python-docx.

## Evaluation
Accuracy, class-wise precision, recall, F1-score, and a confusion matrix are shown for a held-out split of synthetic examples. These results must not be presented as real-world performance. Valid operational evaluation requires approved representative data, expert-reviewed labels, and independent validation.

## Limitations
The sample dataset is synthetic and small; confidence is not calibrated; categories do not establish legal compliance; scanned PDF OCR is not included; indicative terms are not a full explanation.

## Responsible Use
Do not upload confidential or regulated data without authorization and an approved environment. Human review is required before decisions. Operational use would require secure access, retention controls, audit logs, expert validation, and monitoring.

## Future Enhancements
Expand expert-labelled data, add OCR, support policy-specific classes and thresholds, collect reviewer feedback, and add authentication and audit logging.

## Conclusion
The prototype demonstrates document text extraction, risk classification, uncertainty flagging, reviewer support, evaluation, dashboarding, and export for educational use only.
