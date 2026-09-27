# BISync AI

BISync AI is an AI-powered compliance and standards assistant built for the Bureau of Indian Standards (BIS). It allows users to query BIS regulatory documents, standards, and guidelines using natural language retrieval.

> **Project Scope Note:**  
> This repository is a Hackathon Proof of Concept (PoC) / Minimum Viable Product (MVP) built for rapid demonstration. It focuses on validating core Retrieval-Augmented Generation (RAG) capabilities, retrieval accuracy, and interface design using lightweight cloud infrastructure.

---

## Architecture & Tech Stack

- **Frontend:** Single-page editorial interface (HTML5, CSS3, Vanilla JavaScript) hosted on GitHub Pages.
- **Backend API:** Python 3 with FastAPI hosted on Render.
- **Vector Store:** ChromaDB with `sentence-transformers` embeddings for semantic chunk retrieval.
- **Inference Engine:** Groq API (`llama-3.3-70b-versatile`) for low-latency response synthesis.

---

## Key Features

- **Document Ingestion:** Automated PDF text extraction and semantic vector indexing via `ingestion.py`.
- **Grounded Context:** Answers are synthesized directly from indexed BIS reference documents with citations.
- **Low-Latency Synthesis:** Query optimization and vector retrieval powered by Groq LPUs.

---

## Production Roadmap

Planned technical enhancements for enterprise-scale deployment:
- Migration from local ChromaDB to cloud-managed PostgreSQL with `pgvector` (Supabase Vector / AWS Aurora).
- Integration of Role-Based Access Control (RBAC) and BIS SSO authentication.
- Persistent session storage and user interaction analytics.
- Automated document synchronization pipelines connected to official BIS document repositories.

---

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
   cd YOUR_REPO