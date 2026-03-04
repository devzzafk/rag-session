import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Simulation App", layout="wide")

st.title("🔎 RAG Simulation (No LLMs, No APIs)")
st.write("This app demonstrates a simplified Retrieval-Augmented Generation system using TF-IDF and cosine similarity.")

# Load sample document
with open("sample_document.txt", "r") as file:
    document = file.read()

# Option to edit document
text_input = st.text_area("📄 Document Content", value=document, height=250)

# Split into chunks
chunks = [chunk for chunk in text_input.split("\n") if chunk.strip() != ""]

# Vectorization
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

st.subheader("🧠 Ask a Question")
question = st.text_input("Enter your question:")

if question:
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)
    best_match_index = similarities.argmax()
    best_chunk = chunks[best_match_index]
    best_score = similarities[0][best_match_index]

    st.subheader("📌 Most Relevant Chunk Retrieved")
    st.write(best_chunk)

    st.subheader("📊 Similarity Score")
    st.write(round(float(best_score), 4))

    st.subheader("🤖 Final Answer (Simulated Generation)")
    st.write(f"Based on the document, {best_chunk}")
