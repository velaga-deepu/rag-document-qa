# RAG Document Q&A

A simplified retrieval-based document Q&A system using **TF-IDF** and **scikit-learn**.

## 📌 About

This project allows a user to ask questions about a document and retrieves the most relevant passage from the document as the answer.

Instead of generating a new answer using a paid AI API, this project focuses on the **retrieval** part of a RAG system. It uses TF-IDF and cosine similarity to find the passage that is most relevant to the user's question.

This is an **extractive question-answering** approach.

## 🛠️ Technologies Used

* Python
* Scikit-learn
* TF-IDF
* Cosine Similarity

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/velaga-deepu/rag-document-qa.git
cd rag-document-qa
```

Install the required dependency:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

```bash
python rag_document_qa.py
```

The program will ask questions about the sample photosynthesis document.

Example questions:

* What produces oxygen?
* What is chlorophyll?
* How does temperature affect the process?

The system returns the most relevant passage along with its similarity score.

## 💡 Example

**Question:**

> What produces oxygen?

**Retrieved passage:**

> The oxygen produced during photosynthesis is released into the atmosphere as a byproduct.

**Similarity score:** `0.35`

## 🔍 How It Works

1. The document is divided into passages.
2. TF-IDF converts the passages and user's question into numerical vectors.
3. Cosine similarity compares the question with each passage.
4. The passage with the highest similarity is selected.
5. The selected passage and similarity score are displayed.

## 🎯 Project Scope

This project intentionally focuses on the **retrieval** component rather than using a generative AI model.

A future version could be extended with:

* More advanced document chunking
* Top-3 passage retrieval
* Embedding-based semantic search
* Vector databases such as FAISS or Chroma
* Natural-language answer generation
* Support for PDF and other document formats

## 👩‍💻 Author

**Divya Deepika Velaga**

GitHub: [@velaga-deepu](https://github.com/velaga-deepu)
