import streamlit as st
import fitz
from transformers import pipeline
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI PDF Summarizer", layout="centered")
st.title("📄 AI PDF Summarizer")
st.markdown("Upload a PDF and get a concise AI-generated summary.")

# Load summarizer model (use smaller model for faster results if needed)
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    explainier = pipeline("summarization", model="facebook/bart-large-cnn")
    return embedder, explainer

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

embedder, explainer = load_models()
qa_model = load_qa()

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if pdf_file:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    st.subheader("📃 Summary")
    summary = summarizer(full_text[:1000], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    st.write(summary)

    st.subheader("💬 Ask a question about the document")
    question = st.text_input("Your question:")

    if question:
        # Split into chunks and search for best answer
        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
        best_score = -float("inf")
        best_answer = ""
        for chunk in chunks:
            result = qa_model(question=question, context=chunk)
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result['answer']

        st.markdown("**Answer:**")
        st.success(best_answer)