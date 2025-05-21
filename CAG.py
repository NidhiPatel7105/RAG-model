from transformers import pipeline

# Initialize generator
generator = pipeline("text-generation", model="gpt2")

# Known/fixed context
known_context = """
RAG is a technique in NLP that retrieves relevant documents from an external database before answering questions. 
CAG uses a pre-defined or static context to answer questions.
"""

query = "What is CAG used for?"

prompt = f"Context: {known_context}\nQuestion: {query}\nAnswer:"

# Correct: use generator, not 'context'
print(generator(prompt, max_length=100)[0]['generated_text'])
