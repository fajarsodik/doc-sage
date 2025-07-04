import streamlit as st
import fitz
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="DocSage", layout="centered")
st.title("📄 AI PDF Explainer")
st.markdown("Upload a PDF and get a concise AI-generated summary.")

# Load summarizer model (use smaller model for faster results if needed)
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    explainer = pipeline("summarization", model="facebook/bart-large-cnn")
    return embedder, explainer

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

embedder, explainer = load_models()
qa_model = load_qa()

pdf_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])
if pdf_file:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = "".join([page.get_text() for page in doc])

    # Show basic stats
    st.success(f"✅ PDF Loaded: {len(doc)} pages, {len(full_text)} characters.")

    # ---- Full Summary ----
    st.subheader("📝 Summary of the Document")

    summary_chunks = []
    max_chunk_len = 1000  # BART input limit

    for i in range(0, len(full_text), max_chunk_len):
        chunk = full_text[i:i+max_chunk_len]
        result = explainer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summary_chunks.append(result)

    full_summary = " ".join(summary_chunks)
    st.info(full_summary)

    # ---- Semantic Search Setup ----
    st.divider()
    st.subheader("🔍 Ask a Question About a Section")

    chunk_size = 500
    text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # ---- User Query ----
    query = st.text_input("What do you want explained? (e.g. 'Explain the methodology')")
    if query:
        query_embedding = embedder.encode([query])[0]
        D, I = index.search(np.array([query_embedding]), k=1)
        top_chunk = text_chunks[I[0][0]]

        st.subheader("📄 Most Relevant Section")
        st.write(top_chunk)

        with st.spinner("Generating explanation..."):
            explanation = explainer(top_chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

        st.subheader("🧠 Detailed Explanation")
        st.success(explanation)