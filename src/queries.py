import os
import random
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

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

def query_rag(query_text: str, Ollama_model, temperature):
    # Prepare the DB.
    embedding_function = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model=Ollama_model, temperature = temperature)
    response_text = model.invoke(prompt)
    return response_text




# Function to generate queries
def generate_queries(original_query, Ollama_model = "qwen2.5:1.5b"):

    prompt_template = ChatPromptTemplate.from_template(PROMPT_FOR_CREATING_QUERIES)
    prompt = prompt_template.format(original_query=original_query)
    # print(prompt)

    model = Ollama(model=Ollama_model, temperature = 0.7)
    response = model.invoke(prompt)

    generated_queries = response.split("?")
    return generated_queries

# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
        
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

# function to simulate generative output
def generate_output(reranked_results, queries):
    return f"Final output based on and reranked documents: {list(reranked_results.keys())}"

def query_with_rag_fusion(original_query, Ollama_model, temperature):
    generated_queries = generate_queries(original_query, Ollama_model)
    
    all_results = {}
    embedding_function = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    for query in generated_queries:
        selected_docs = db.similarity_search_with_score(query, k=5)
        scores = {doc[0].page_content: round(random.uniform(0.7, 0.9), 2) for doc in selected_docs}
        search_results = {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}
        all_results[query] = search_results
    
    reranked_results = reciprocal_rank_fusion(all_results)
    
    final_output = generate_output(reranked_results, generated_queries)
    
    return final_output