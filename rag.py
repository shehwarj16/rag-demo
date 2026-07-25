import os
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# ---- SETUP ----
print("Loading embedding model... (first time takes a minute)")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="company_docs")

client = Anthropic(api_key="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")

# ---- STEP 1: LOAD DOCUMENTS ----
docs_folder = "docs"
documents = []
filenames = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(docs_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
            filenames.append(filename)

print(f"Loaded {len(documents)} documents: {filenames}")

# ---- STEP 2: CREATE EMBEDDINGS AND STORE ----
embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=filenames
)

print("Documents embedded and stored.\n")

# ---- STEP 3: ASK A QUESTION ----
question = input("Ask a question about your documents: ")

question_embedding = embedder.encode([question]).tolist()

results = collection.query(
    query_embeddings=question_embedding,
    n_results=2  # top 2 most relevant chunks
)

retrieved_chunks = results['documents'][0]
retrieved_files = results['ids'][0]

print(f"\nRetrieved from: {retrieved_files}")

# ---- STEP 4: SEND TO CLAUDE ----
context = "\n\n".join(retrieved_chunks)

prompt = f"""Answer the question using ONLY the information below. If the answer isn't in the information provided, say so.

INFORMATION:
{context}

QUESTION: {question}
"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- ANSWER ---")
print(response.content[0].text)