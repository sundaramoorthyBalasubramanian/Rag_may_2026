import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "vector_db_data"

# -----------------------------
# Helper: clean + validate chunks
# -----------------------------
def filter_chunks(chunks):
    filtered = []

    for chunk in chunks:
        text = chunk.page_content.strip()

        # Skip empty / very small chunks
        if len(text) < 50:
            continue

        filtered.append(chunk)

    print(f"Filtered chunks: {len(filtered)} / {len(chunks)} kept")
    return filtered


# -----------------------------
# Save to Vector DB
# -----------------------------
import os
import shutil
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "vector_db_data"


# -----------------------------
# Better Deduplication
# -----------------------------
def normalize_text(text):
    # remove extra spaces + lowercase
    return " ".join(text.lower().split())


def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()


def deduplicate_chunks(chunks):
    seen_hashes = set()
    unique_chunks = []

    for chunk in chunks:
        text = chunk.page_content.strip()

        # normalize first
        normalized = normalize_text(text)

        # skip tiny chunks
        if len(normalized.split()) < 20:
            continue

        h = hash_text(normalized)

        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_chunks.append(chunk)

    print(f"🧹 Deduplicated: {len(unique_chunks)} / {len(chunks)} kept")
    return unique_chunks


# -----------------------------
# Save DB
# -----------------------------
def save_to_db(chunks, rebuild=True):

    # 🔥 IMPORTANT: clear old DB
    if rebuild and os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("🗑️ Old vector DB deleted")

    # 1. Deduplicate
    chunks = deduplicate_chunks(chunks)

    # 2. Filter bad chunks (your existing function)
    chunks = filter_chunks(chunks)

    # 3. Embedding model (upgrade recommended)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",   # 🔥 better retrieval
        model_kwargs={"device": "cpu"}
    )

    # 4. Create DB directory
    os.makedirs(DB_DIR, exist_ok=True)

    # 5. Store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="my_rag_collection"
    )

    print(f"✅ Stored {len(chunks)} chunks in '{DB_DIR}'")

    # 6. Debug sample
    if chunks:
        print("\n--- Sample Stored Chunk ---")
        print("Text:", chunks[0].page_content[:200])
        print("Metadata:", chunks[0].metadata)

    return vector_db


# -----------------------------
# Load DB (for querying later)
# -----------------------------
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"}
    )

    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="my_rag_collection"
    )


    print("@sundar Vector DB loaded from disk")
    return vector_db

def build_context(docs):
    context = ""

    for i, doc in enumerate(docs):
        context += f"\n[Chunk {i+1}]\n"
        context += doc.page_content + "\n"

    return context

def build_prompt(query, context):
    prompt = f"""
    You are a helpful AI assistant.

    Use ONLY the provided context to answer the question.
    If the answer is not in the context, say "Not found in document".

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    return prompt    

from openai import OpenAI

client = OpenAI()

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # fast + cheap
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def rag_query(db, query):
    results = db.similarity_search(query, k=5)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
        print("Metadata:", doc.metadata)

    
    context = build_context(results)
    prompt = build_prompt(query, context)

    answer = ask_llm(prompt)

    return answer

# -----------------------------
# Test Retrieval (VERY IMPORTANT)
# -----------------------------
def test_retrieval(vector_db, query="What is the model architecture?"):
    results = vector_db.similarity_search(query, k=3)

    print("\n===== RETRIEVAL TEST =====")
    print("Query:", query)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
        print("Metadata:", doc.metadata)
    return results    