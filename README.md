# Local LLaMA Chatbot

A simple offline chatbot using **LLaMA 3B** and **Ollama**, with **Streamlit UI** for interaction — now extended with a **FAISS-based vector database retriever** for Retrieval-Augmented Generation (RAG).

## Features
- Text-based conversation
- Multi-turn chat using Streamlit `session_state`
- Runs locally without external servers
- **RAG pipeline**: retrieves relevant context from a local knowledge base using FAISS vector similarity search before generating a response

## How to Run
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Make sure Ollama is running locally with the llama3.2:3b model pulled:
```
ollama pull llama3.2:3b
```

3. Run the chatbot:
```
streamlit run app.py
```

## Project Structure
```
├── app.py              # Streamlit chatbot UI + RAG integration
├── retriever.py         # FAISS vector database retriever module
├── docs/
│   └── knowledge.txt    # Knowledge base used for retrieval (edit with your own notes)
├── requirements.txt
└── pdf.html             # Separate utility: PDF-to-Excel upload form (n8n workflow, not part of the chatbot)
```

## How RAG works here
1. `retriever.py` loads text chunks from `docs/knowledge.txt` and converts them into vectors (TF-IDF)
2. Vectors are indexed in FAISS for fast similarity search
3. When you send a message, the top relevant chunks are retrieved and added as context to the prompt sent to Ollama
4. The retrieved context is shown in an expandable section under each response
