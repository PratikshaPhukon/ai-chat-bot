import os
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


try:
    # --- Load Lessons ---
    print("📂 Loading lessons from data/lessons.json...", flush=True)
    with open("data/lessons.json", "r", encoding="utf-8") as f:
        lessons = json.load(f)
    print("✅ Reached after loading JSON", flush=True)
    print(f"✅ Loaded {len(lessons)} lessons.", flush=True)

    # --- Chunking ---
    print("✂️ Splitting text into chunks...", flush=True)
    texts, metadatas = [], []
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    for title, content in lessons.items():
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"lesson": title, "chunk_index": i})

    print(f"🧩 Total chunks: {len(texts)}", flush=True)
    print(f"📌 Example chunk: {texts[0][:100]}...", flush=True)

    # --- Embeddings ---
    print("🔧 Initializing embedding model...", flush=True)
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    # --- FAISS Vectorstore ---
    persist_dir = "faiss_index"
    os.makedirs(persist_dir, exist_ok=True)
    print("📦 Creating and saving FAISS vectorstore...", flush=True)
    vectordb = FAISS.from_texts(texts, embedding=embed_model, metadatas=metadatas)
    vectordb.save_local(persist_dir)
    print(f"✅ FAISS vectorstore saved at: {persist_dir}", flush=True)

    # --- Reload and Test ---
    print("🔁 Reloading FAISS for test...", flush=True)
    test_db = FAISS.load_local(persist_dir, embeddings=embed_model, allow_dangerous_deserialization=True)

    print("🔍 Running similarity search for: 'What is cyber law?'", flush=True)
    results = test_db.similarity_search("What is cyber law?", k=1)

    if results:
        print("🔍 Top result preview:", results[0].page_content[:150], flush=True)
    else:
        print("⚠️ No results found.", flush=True)

    print("✅ Vectorstore built and saved to:", persist_dir, flush=True)

except Exception as e:
    print("❌ An error occurred:", e, flush=True)
