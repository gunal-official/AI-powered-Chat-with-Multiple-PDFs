# 📄 AI-Powered Document Assistant

Chat with your PDFs using RAG (Retrieval-Augmented Generation). Upload one or
more PDFs and ask questions — answers stream in real time and cite the exact
file and page they came from.

## ✨ Features

- **Multi-PDF chat** — upload several PDFs and query them all at once
- **Streaming answers** — tokens appear as they're generated
- **Source citations** — every answer shows which file & page it used
- **Conversation memory** — follow-up questions understand context
- **Smart caching** — PDFs are only embedded once, not on every interaction
- **Sidebar controls** — model picker, temperature, retrieval depth, clear chat

## 🏗️ How it works

```
PDFs ──▶ PyPDFLoader ──▶ chunking (RecursiveCharacterTextSplitter)
                              │
                              ▼
                   HuggingFace embeddings (all-MiniLM-L6-v2)
                              │
                              ▼
                        FAISS vector store
                              │
   question ──▶ retrieve top-k chunks ──▶ Groq LLM ──▶ streamed answer + citations
```

| Component    | Technology                              |
| ------------ | --------------------------------------- |
| UI           | Streamlit                                |
| LLM          | Groq (Llama 3.1 / 3.3, GPT-OSS)          |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (in-memory)                        |
| Orchestration| LangChain                                |

## 🚀 Getting started

### 1. Clone & install

```bash
git clone https://github.com/gunal-official/AI-Powered-Document-Assistant.git
cd AI-Powered-Document-Assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Get a Groq API key

Create a free key at [console.groq.com](https://console.groq.com) and either:

- put it in a `.env` file in the project root:

  ```env
  GROQ_API_KEY=your_key_here
  ```

- **or** paste it into the sidebar when the app is running.

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload PDFs, and start asking questions.

## 📁 Project structure

```
├── app.py            # Streamlit UI + chat loop
├── utils.py          # PDF loading, chunking, embeddings, FAISS
├── requirements.txt
└── README.md
```

## 🗺️ Roadmap

- [ ] More file types (DOCX, TXT, Markdown, CSV)
- [ ] OCR fallback for scanned PDFs
- [ ] Hybrid retrieval (BM25 + vectors) and re-ranking
- [ ] Persistent vector store across sessions
- [ ] One-click document summarization
- [ ] Chat export (Markdown / PDF)
