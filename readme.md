# 🔍 LLM-Free AI Fact Checker using BART + FAISS + Keyword Matching

This project is a lightweight AI-powered fact-checking system that allows users to verify the truthfulness of short public statements (e.g., news or social media posts) by comparing them against a database of verified facts.

Real life use cases:- To check whether any news is fake or real especially during the pandemic,war situation to avoid mishaps this could be effectively used.

⚡ Powered by:
-  BART for zero-shot classification
-  MiniLM embeddings for semantic similarity
-  FAISS for fast vector search
-  Streamlit for user interface

---

## 🧠 Core Features

| Component              | Tool Used                                     |
|------------------------|-----------------------------------------------|
| **Claim Extraction**   | spaCy NER + fallback to full statement         |
| **Embedding**          | SentenceTransformer (all-MiniLM-L6-v2)         |
| **Semantic Search**    | FAISS                                          |
| **Verification**       | Zero-Shot Classification (`facebook/bart-large-mnli`) |
| **Backup Method**      | Simple keyword matching & negation detection  |
| **UI**                 | Streamlit Web App                             |

---

## 🚀 How It Works

1. **User enters a news statement** or uploads a `.txt` file.
2. The app extracts core claims/entities.
3. It retrieves top similar verified facts from a local database using vector embeddings.
4. Using either:
   - 🤖 BART-based AI classification, or
   - 🧩 Simple keyword-based heuristics,
   it determines if the claim is ✅ True, ❌ False, or 🟡 Unverifiable.
5. Shows verdict, matched evidence, and technical reasoning.
6. streamlit run app.py

---





