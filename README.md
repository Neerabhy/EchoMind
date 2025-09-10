# 🧠 EchoMind  
An advanced **AI-powered conversational assistant** built on a **dual-model architecture** that combines **RAG-based retrieval**, **intelligent memory management**, and **automatic web search** to deliver **context-aware**, **precise**, and **low-latency** responses.

---

## 🚀 Features  

### ⚡ Dual-Model Architecture  
- **Main Model** → Handles reasoning, generates final answers, integrates context, and decides when to fetch external data.  
- **Retriever Model** → Lightweight, optimized for **RAG-based retrieval** and **precise summarization** only.  

### 🧠 Intelligent Memory System  
- Summarizes conversations **on-the-fly** for long-context retention.  
- Uses **turn relevance classification** to avoid storing redundant or irrelevant data.  
- Retrieves memory **only when the main model detects missing details**.  

### 🔍 Automatic Web Search  
- If knowledge is outdated or missing, the system **auto-triggers a web search**.  
- Integrates summarized search results **seamlessly into responses**.  

### 🎯 Optimized RAG  
- Handles **QA-formatted memory** effectively.  
- Reduces hallucinations using **AI-enhanced retrieval pipelines**.  

### 📊 Benchmarking Suite  
- Compare chatbot performance with **LLaMA3-70B** or other LLMs.  
- Uses an **AI Judge** (via **Groq API**) to score responses based on:
  - **Context Retention**  
  - **Fluency**  
  - **Accuracy**  
  - **Grammar**  

---

## 🛠️ Tech Stack  

- **Language**: Python 3.10+  
- **Framework**: FastAPI  
- **Models**:  
  - **Main LLM** → LLaMA3 / Mixtral via Groq API  
  - **Retriever** → Lightweight RAG + summarization  
- **Vector Database**: Sentence Transformers embeddings  
- **APIs**: Groq API + optional Web Search APIs  
- **Benchmarking**: ROUGE, BLEU, AI Judge-based scoring  

---
