from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings