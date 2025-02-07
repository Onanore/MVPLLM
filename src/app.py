import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import ollama
import chromadb
import os

# Constants
CHUNK_SIZE = 1000  # Size of text chunks for processing
DEFAULT_MODEL = "deepseek-r1:1.5b" # Default language model
DEFAULT_TEMPERATURE = 0.4 # Default temperature for language model (controls randomness)
DEFAULT_MAX_TOKENS = 2048 # Default maximum number of tokens for language model output
DB_DIR = "chroma_db"  # Directory to store the ChromaDB database

def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Initialize message history for the chat
    if "collection" not in st.session_state:
        # Create the database directory if it doesn't exist
        os.makedirs(DB_DIR, exist_ok=True)
        
        # Initialize ChromaDB client, specifying the storage path
        client = chromadb.PersistentClient(path=DB_DIR)
        
        # Create or retrieve the document collection
        st.session_state.collection = client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Set the distance metric for similarity search
        )

def initialize_streamlit():
    st.set_page_config(page_title="Ollama Chatbot", layout="wide") # Set page title and layout
    st.title("🤖 Local Chatbot with Ollama") # Set the title of the app
    st.markdown("Upload a file and ask questions about its content.") # Provide instructions

def create_sidebar():
    with st.sidebar: # Create a sidebar for settings
        models = ["deepseek-r1:1.5b", "deepseek-r1", "qwen2.5:1.5b"] # Available language models
        model = st.selectbox("Model:", models, index=0) # Model selection dropdown
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE) # Temperature slider
        max_tokens = st.slider("Maximum Tokens", 1024, 32000, DEFAULT_MAX_TOKENS) # Max tokens slider
        uploaded_file = st.file_uploader(
            "Upload a file (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"] # Allowed file types
        )
        return model, temperature, max_tokens, uploaded_file # Return selected settings

def process_file(file):
    try:
        text = ""
        if file.type == "application/pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() # Extract text from each page
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs]) # Extract text from each paragraph
        elif file.type == "text/plain":
            text = file.read().decode("utf-8") # Decode text from plain text file
        else:
            raise ValueError(f"Unsupported file type: {file.type}") # Raise error for unsupported types
        return text
    except Exception as e:
        st.error(f"Error processing file: {str(e)}") # Display error message
        return None

def split_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] # Split text into chunks

def generate_embedding(text):
    try:
        response = ollama.embeddings(
            model="nomic-embed-text", # Model for generating embeddings
            prompt=text
        )
        return response["embedding"] # Return the generated embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}") # Display error message
        return None

def update_collection(chunks):
    try:
        # Get all existing document IDs
        all_ids = st.session_state.collection.get()["ids"]
        
        # If documents exist, delete them
        if all_ids:
            st.session_state.collection.delete(
                ids=all_ids
            )
        
        # Add new chunks
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            if embedding:
                st.session_state.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[f"chunk_{i}"] # Assign unique IDs to chunks
                )
        return True
    except Exception as e:
        st.error(f"Error updating database: {str(e)}") # Display error message
        return False

def get_context(prompt):
    try:
        query_embedding = generate_embedding(prompt)
        if query_embedding:
            results = st.session_state.collection.query(
                query_embeddings=[query_embedding],
                n_results=3 # Retrieve top 3 most similar chunks
            )
            return "Context from the document:\n" + "\n".join([f"- {doc}" for doc in results["documents"][0]]) # Format context
        return ""
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}") # Display error message
        return ""

def generate_response(prompt, context, model, temperature, max_tokens):
    try:
        response = ollama.generate(
            model=model,
            prompt=f"{context}\n\nQuestion: {prompt}", # Combine context and prompt
            stream=False, # Disable streaming for simpler response handling
            options={
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        return response["response"] # Return the generated response
    except Exception as e:
        return f"Error: {str(e)}. Make sure Ollama is running."

def main():
    initialize_streamlit()
    initialize_session()

    model, temperature, max_tokens, uploaded_file = create_sidebar() # Get settings from sidebar

    if uploaded_file:
        file_text = process_file(uploaded_file) # Process uploaded file
        if file_text:
            chunks = split_text(file_text) # Split text into chunks
            if update_collection(chunks): # Update the database
                st.success("File successfully uploaded and processed!") # Display success message

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # Display messages in chat bubbles
            st.markdown(message["content"])

    # Handle user input
    prompt = st.chat_input("Your question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt}) # Add user message to history
        with st.chat_message("user"):
            st.markdown(prompt) # Display user message

        context = get_context(prompt) # Get relevant context
        answer = generate_response(prompt, context, model, temperature, max_tokens) # Generate response

        st.session_state.messages.append({"role": "assistant", "content": answer}) # Add assistant message to history
        with st.chat_message("assistant"):
            st.markdown(answer) # Display assistant message

if __name__ == "__main__":
    main()