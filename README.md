# MVPLLM

A streamlit-based chatbot application that uses Ollama for local language model inference and RAG (Retrieval Augmented Generation) capabilities. The application supports document upload, similarity search, and RAG Fusion for improved response quality.

## Features

- 🤖 Local LLM inference using Ollama
- 📄 Document upload and processing (TXT files)
- 🔍 RAG (Retrieval Augmented Generation) implementation
- 🔄 RAG Fusion support for better search results
- 💾 ChromaDB for vector storage
- 🎯 Adjustable temperature and model selection
- 🎨 Clean Streamlit UI

## Installation

1. First, make sure you have [Ollama](https://ollama.ai/) installed on your system.

2. Clone the repository:
```bash
git clone https://github.com/Onanore/MVPLLM.git
cd MVPLLM
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Pull the required Ollama models:
```bash
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Use the sidebar to:
   - Select your preferred model
   - Adjust the temperature
   - Upload your text documents
   - Toggle RAG Fusion

4. Start chatting with the bot about your uploaded documents!

## Demo

Here's how the application looks and works:

<img src="/api/placeholder/800/400" alt="Ollama Chatbot Demo Screenshot" />

## Examples

1. **Basic RAG Query**
   - Upload a document
   - Ask questions about the content
   - Get responses based on the document context

2. **RAG Fusion Enhanced Search**
   - Enable RAG Fusion in the sidebar
   - Ask complex questions
   - Get responses based on multiple search queries and reranked results

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── database.py         # Document processing and ChromaDB operations
├── embedding.py        # Embedding model configuration
├── queries.py          # RAG and RAG Fusion query processing
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Notes

- The application uses Ollama for local inference, ensuring privacy and reducing API costs
- RAG Fusion implementation helps improve the quality of responses by generating multiple search queries
- Document chunking is optimized for better context retrieval
- The application currently supports TXT files, with plans to add support for more formats
