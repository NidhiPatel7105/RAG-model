import os
import fitz  # PyMuPDF
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Step 1: Load and process PDF documents ===

pdf_folder = r"C:\Users\nidhi\OneDrive\Desktop\Task-1\docs\2.3_Precinmac Capex ROI Examples"  # Folder with PDFs extracted from ZIP
docs = []

print("Extracting text from PDFs...\n")

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        docs.append(text)

if not docs:
    print("❌ No supported documents found in 'docs' folder.")
    exit()

print(f"✅ Loaded {len(docs)} documents!\n")

# === Step 2: Set device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}\n")

# === Step 3: Load Text Generation Model ===
generator = pipeline(
    "text-generation",
    model="gpt2",
    pad_token_id=50256,
    device=0 if device == "cuda" else -1,
)

# === Step 4: QA Loop ===
print("RAG model is ready! Type your question below.\n(Type 'exit' to quit)\n")

while True:
    query = input("Enter your question: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    # TF-IDF similarity for retrieval
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(docs + [query])
    cosine_sim = cosine_similarity(doc_vectors[-1], doc_vectors[:-1])
    top_doc = docs[cosine_sim.argmax()]

    # Truncate context to 1000 characters max
    short_context = top_doc[:1000]

    prompt = f"Context: {short_context}\nQuestion: {query}\nAnswer:"

    try:
        result = generator(prompt, max_new_tokens=150, do_sample=True, truncation=True)[0]["generated_text"]
        print("\n--- Answer ---\n" + result.split("Answer:")[-1].strip() + "\n")
    except Exception as e:
        print("⚠️ Error during generation:", e)
