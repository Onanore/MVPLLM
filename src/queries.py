from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.load import dumps, loads

from database import *

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

PROMPT_FOR_CREATING_QUERIES = """
You are a helpful assistant that generates multiple search queries based on a single input query.

Generate multiple search queries related to: {original_query}

OUTPUT (4 queries):
"""

def ask_ai(query_text, context_text, Ollama_model, temperature):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model=Ollama_model, temperature = temperature)
    response_text = model.invoke(prompt)

    return response_text


def query_rag(query_text: str, Ollama_model, temperature):
    # Initialize embedding function and Chroma vector store.
    embedding_function = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the retrieved documents into a single context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Generate the final response
    response_text = ask_ai(query_text, context_text, Ollama_model, temperature)

    return response_text




# Function to generate queries
def generate_queries(original_query, Ollama_model = "qwen2.5:1.5b"):

    prompt_template = ChatPromptTemplate.from_template(PROMPT_FOR_CREATING_QUERIES)
    prompt = prompt_template.format(original_query=original_query)

    model = Ollama(model=Ollama_model, temperature = 0.4)
    response = model.invoke(prompt)

    generated_queries = [q.strip() for q in response.split('\n') if q.strip()]
    return generated_queries



def query_with_rag_fusion(original_query, Ollama_model, temperature):
    # Initialize embedding function and Chroma vector store
    embedding_function = get_embedding()
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(search_type="mmr")

    # Generate multiple queries
    genQ = generate_queries(original_query)

    # Retrieve relevant documents for each generated query
    retrieval_results = []
    for query in genQ:
        results = retriever.invoke(query)
        retrieval_results.append(results)

    # Retrieve documents for the original query
    retrieval_results.append( retriever.invoke({"question": original_query}))

    # Deduplicate retrieved documents across all queries
    lst=[]
    for ddxs in retrieval_results:
        for ddx in ddxs:
            if ddx.page_content not in lst:
                lst.append(ddx.page_content)

    # Rerank the retrieved documents using Reciprocal Rank Fusion (RRF)
    reranked_results = reciprocal_rank_fusion(retrieval_results)

    # Format the retrieved documents into a single context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in reranked_results])

    # Generate the final response
    response_text = ask_ai(original_query, context_text, Ollama_model, temperature)
    return response_text


def reciprocal_rank_fusion(results):
    fused_scores = {}
    k=60
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # final reranked result
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results