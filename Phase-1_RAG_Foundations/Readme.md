# 📘 Phase 1 – RAG Foundations (Weeks 1–2)

This folder is part of my **RAG Specialization Journey**.  
Here, I’m focusing on building strong foundations by understanding how Retrieval-Augmented Generation (RAG) works end-to-end and by setting up my first working pipeline.  

🎯 **Goal:** Develop a solid understanding of RAG concepts and implement a basic but functional RAG system.

---

## 🚀 What is RAG & Why it Matters
Retrieval-Augmented Generation (RAG) combines **retrieval** from a knowledge base with **generation** from a Large Language Model (LLM).  

- Provides **fact-grounded answers**  
- Reduces **hallucinations**  
- Scales knowledge without retraining the model  

---

## 🔎 Embeddings Overview
- **Dense embeddings** → capture semantic meaning (e.g., OpenAI, HuggingFace).  
- **Sparse embeddings** → keyword-based retrieval (e.g., BM25).  

---

## 📐 Chunking Strategies
- **Fixed size**: e.g., 500 tokens per chunk  
- **Overlap**: sliding window chunks for context preservation  
- **Semantic chunking**: splits by sections/meaning  

👉 In this phase, I’ll experiment with multiple strategies to compare retrieval quality.  

---

## 🗄️ Vector Database Options
I’m exploring both **local** and **production-grade** options:  
- [FAISS](https://github.com/facebookresearch/faiss) – lightweight, local testing  
- [Qdrant](https://qdrant.tech/) – scalable and production-ready  
- [Milvus](https://milvus.io/) – distributed system  
- [Weaviate](https://weaviate.io/) – hybrid search  

---

## ⚡ LangChain Basics
Key components I’ll be working with:  
- **Documents & Loaders** (ingesting data)  
- **TextSplitters** (chunking text)  
- **Retrievers** (search over embeddings)  
- **Chains** (retrieval + LLM combined)  


---

## 📚 Resources I’m Using
- [LangChain Docs](https://python.langchain.com/docs/)  
- [Qdrant Docs](https://qdrant.tech/documentation/)  
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)  
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)  

---

## 🛠️ Hands-on Experiments
- [ ] 
- [ ]
- [ ]   
- [ ]   

---

## 📌 Phase 1 Deliverables
By the end of this phase, I aim to have:  
- ✅ A working **RAG prototype (FAISS + LLM)**  
- ✅ Notes/Notebooks on **chunking experiments**  
- ✅ A **mini Streamlit app** for demo purposes  

---

💡 *This phase sets the foundation for advanced RAG techniques I’ll cover in later phases.*  
