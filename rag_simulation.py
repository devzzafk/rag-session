from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load document
with open("sample_document.txt", "r") as file:
    document = file.read()

# Step 2: Split into chunks
chunks = document.split("\n")

print("Document Chunks:")
for i, chunk in enumerate(chunks):
    print(f"{i+1}. {chunk}")

# Step 3: Convert text to embeddings using TF-IDF
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# Step 4: Ask question
question = input("\nAsk a question: ")

# Convert question to embedding
question_vector = vectorizer.transform([question])

# Step 5: Compute similarity
similarities = cosine_similarity(question_vector, chunk_vectors)

# Step 6: Find best matching chunk
best_match_index = similarities.argmax()
best_chunk = chunks[best_match_index]

print("\nMost Relevant Chunk Found:")
print(best_chunk)

# Step 7: Simulated "Generation"
print("\nFinal Answer:")
print(f"Based on the document, {best_chunk}")
