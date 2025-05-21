from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Step 1: Some documents (like a knowledge base)
docs = [
    "RAG stands for Retrieval-Augmented Generation. It retrieves relevant documents before generating an answer.",
    "CAG uses a fixed context or document to answer user questions without dynamic retrieval.",
    "RAG uses embeddings and vector search to retrieve information.",
]

# Step 2: User query
query = "What is RAG in NLP?"

# Step 3: Retrieve relevant docs using cosine similarity
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs + [query])
cosine_sim = cosine_similarity(doc_vectors[-1], doc_vectors[:-1])
top_doc = docs[cosine_sim.argmax()]

# Step 4: Generate answer using LLM (here using GPT2 for offline demo)
generator = pipeline("text-generation", model="gpt2")
prompt = f"Based on this context: {top_doc}\nAnswer the question: {query}\nAnswer:"

print(generator(prompt, max_length=100)[0]['generated_text'])
