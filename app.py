import io, re
from pathlib import Path
import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI Document Risk & Compliance Classifier", page_icon="🛡️", layout="wide")
DATA = Path(__file__).parent / "data" / "sample_documents.csv"
LABELS = ["Low", "Medium", "High"]
THRESHOLD = 0.60

@st.cache_data
def load_data():
    return pd.read_csv(DATA).dropna(subset=["text", "risk_level"])

@st.cache_resource
def train_model():
    data = load_data()
    xtr, xte, ytr, yte = train_test_split(data.text, data.risk_level, test_size=.25, random_state=42, stratify=data.risk_level)
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=5000)),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(xtr, ytr)
    return model, xte, yte, model.predict(xte)

def extract(file):
    raw = file.getvalue()
    name = file.name.lower()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
    raise ValueError("Unsupported file type")

def explain(model, text, label):
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    row = vec.transform([text])
    idx = list(clf.classes_).index(label)
    contributions = row.toarray()[0] * clf.coef_[idx]
    terms = vec.get_feature_names_out()
    best = [terms[i] for i in contributions.argsort()[::-1] if contributions[i] > 0 and row[0,i] > 0][:5]
    return ", ".join(best) if best else "No strong indicative terms"

def classify(model, name, text):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) < 20:
        return {"Document":name, "Risk level":"Needs review", "Confidence":0.0, "Manual review":"Yes", "Key terms":"Insufficient extracted text"}
    probs = model.predict_proba([text])[0]
    i = int(probs.argmax())
    label, conf = str(model.classes_[i]), float(probs[i])
    return {"Document":name, "Risk level":label, "Confidence":round(conf,3),
            "Manual review":"Yes" if conf < THRESHOLD else "No", "Key terms":explain(model,text,label)}

st.title("🛡️ AI Document Risk & Compliance Classifier")
st.caption("Classify business documents by risk level and flag uncertain results for human review.")
st.info("Educational prototype only. Review all results with a qualified person; this is not a legal or compliance determination.")
with st.sidebar:
    st.header("Settings")
    st.metric("Manual-review threshold", f"{THRESHOLD:.0%}")
    st.caption("Predictions below this score are flagged.")
    st.write("Model: TF-IDF + Logistic Regression")
    st.write("Data: synthetic examples")

try:
    model, x_test, y_test, pred = train_model()
    data = load_data()
except Exception as e:
    st.error(f"Model setup failed: {e}")
    st.stop()

analyze, dashboard, performance, about = st.tabs(["Analyze documents","Dashboard","Model performance","About"])
with analyze:
    st.subheader("Upload or paste documents")
    uploads = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","txt"], accept_multiple_files=True)
    pasted = st.text_area("Or paste non-confidential sample text", height=160)
    if st.button("Classify documents", type="primary"):
        docs = []
        for f in uploads or []:
            try: docs.append((f.name, extract(f)))
            except Exception as e: st.warning(f"{f.name}: {e}")
        if pasted.strip(): docs.append(("Pasted document",pasted))
        if not docs: st.warning("Upload a document or paste text first.")
        else:
            st.session_state["results"] = [classify(model,n,t) for n,t in docs]
            st.session_state["texts"] = {n:re.sub(r"\\s+"," ",t or "").strip() for n,t in docs}
    rows = st.session_state.get("results",[])
    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Classification results")
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.caption("Confidence is not a calibrated probability. Indicative terms are not a complete explanation.")
        c1,c2 = st.columns(2)
        c1.download_button("Download CSV",df.to_csv(index=False).encode(), "document_risk_results.csv","text/csv",use_container_width=True)
        c2.download_button("Download JSON",df.to_json(orient="records",indent=2).encode(),"document_risk_results.json","application/json",use_container_width=True)
        for i,row in enumerate(rows):
            with st.expander(f"Review: {row['Document']}"):
                st.write(f"**Risk:** {row['Risk level']} · **Confidence:** {row['Confidence']:.1%} · **Manual review:** {row['Manual review']}")
                st.write("**Indicative terms:**",row["Key terms"])
                st.text_area("Extracted text",st.session_state.get("texts",{}).get(row["Document"],""),height=100,disabled=True,key=f"txt_{i}")
with dashboard:
    rows=st.session_state.get("results",[])
    if not rows: st.info("Classify documents to populate the dashboard.")
    else:
        df=pd.DataFrame(rows)
        a,b,c,d=st.columns(4)
        a.metric("Documents",len(df)); b.metric("High risk",int((df["Risk level"]=="High").sum()))
        c.metric("Flagged for review",int((df["Manual review"]=="Yes").sum()))
        d.metric("Average confidence",f"{df['Confidence'].mean():.1%}")
        st.bar_chart(df["Risk level"].value_counts().reindex(LABELS+["Needs review"],fill_value=0))
with performance:
    st.subheader("Held-out synthetic-data evaluation")
    st.metric("Accuracy",f"{accuracy_score(y_test,pred):.1%}")
    st.caption("Metrics describe only the included synthetic examples and do not establish real-world performance.")
    rep=pd.DataFrame(classification_report(y_test,pred,labels=LABELS,output_dict=True,zero_division=0)).T
    st.dataframe(rep.round(3),use_container_width=True)
    st.write("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test,pred,labels=LABELS),index=LABELS,columns=LABELS),use_container_width=True)
with about:
    st.markdown("**Workflow:** document text extraction → text cleaning → TF-IDF features → Logistic Regression → confidence threshold → human review → dashboard and export.")
    st.warning("The dataset is synthetic. Do not use this prototype for real compliance decisions or upload confidential/regulated documents without authorization and an approved environment.")
    st.write(f"Training examples: {len(data)}")
