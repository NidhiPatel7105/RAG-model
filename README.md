# 📄 PDF-Based RAG QA System using GPT-2

This project is a simple **Retrieval-Augmented Generation (RAG)** pipeline that answers user questions based on a set of PDF documents. It combines classical IR techniques like **TF-IDF** for retrieval with **GPT-2** for answer generation.

---

## 🚀 Features

- ✅ Extracts text from multiple PDFs using PyMuPDF
- ✅ Uses **TF-IDF + Cosine Similarity** for relevant context retrieval
- ✅ Generates context-aware answers using Hugging Face's **GPT-2**
- ✅ Runs on **CPU or GPU**
- ✅ Command-line based interactive QA loop

---

## 🧠 How It Works

1. **Document Ingestion**: All PDFs inside the `docs/` folder are read and their content extracted.
2. **User Input**: You can ask a natural language question via the CLI.
3. **Retrieval Step**: TF-IDF ranks the PDFs based on relevance to your question.
4. **Generation Step**: GPT-2 generates an answer based on the retrieved content.

---

## 🧰 Requirements

Install the following packages via pip:

```bash
pip install torch transformers scikit-learn pymupdf
    
📁 Project Structure
bash
Copy
Edit
Task-1/
│
├── docs/                              # Folder containing your input PDF files
│   └── *.pdf
│
├── RAG_model.py                       # Main script
├── README.md                          # This file
🖥️ Usage
Place your PDF documents inside the docs/ folder.

Run the model:

bash
Copy
Edit
python RAG_model.py
Start asking questions! Example:

csharp
Copy
Edit
Enter your question: What is the ROI in the examples?
To exit:

bash
Copy
Edit
Enter your question: exit
⚙️ Configuration
The script will automatically detect if a GPU is available. If not, it will run on the CPU.

You can change the folder path in RAG_model.py here:

python
Copy
Edit
pdf_folder = r"C:\Users\nidhi\OneDrive\Desktop\Task-1\docs\2.3_Precinmac Capex ROI Examples"
📌 Notes
The maximum context size for GPT-2 is limited, so only the first 1000 characters of the most relevant document are used.

GPT-2 is not fine-tuned for question answering, so answers may be general. For better accuracy, consider using a model like distilbert-base-cased-distilled-squad.
