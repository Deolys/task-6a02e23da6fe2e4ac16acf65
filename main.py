"""
RAG Agent with Qdrant and Ollama
Author: Auto-generated for task 6a02e23da6fe2e4ac16acf65
"""
import os
from pathlib import Path
from typing import List, Dict

from langchain_ollama import Ollama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import HumanMessage

# Initialize embeddings and LLM
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = Ollama(model="llama3", temperature=0)

# Vector store setup (Qdrant local instance assumed running on default localhost:6333)
vector_store = QdrantVectorStore(
    client=qdrant_client.QdrantClient(url="http://localhost:6333"),
    embeddings=embeddings,
    collection_name="rag_collection",
)

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

@tool("search_knowledge_base")
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a semantic search in the knowledge base.
    Returns a list of documents with their metadata and score.
    """
    results = vector_store.similarity_search_with_score(query, k=max_results)
    return [{"content": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in results]

@tool("add_to_knowledge_base")
def add_to_knowledge_base(content: str, title: str = "") -> str:
    """
    Add a new document to the knowledge base.
    The content is split into chunks before being stored.
    Returns the number of chunks added.
    """
    docs = text_splitter.split_text(content)
    vector_store.add_documents([{"page_content": d, "metadata": {"title": title}} for d in docs])
    return f"Added {len(docs)} chunks to the knowledge base."

# Agent setup
SYSTEM_PROMPT = (
    "You are an AI assistant that can search and add documents to a local RAG knowledge base using Qdrant and Ollama embeddings."
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=create_openai_functions_agent(llm=llm, tools=[search_knowledge_base, add_to_knowledge_base], system_message=SYSTEM_PROMPT),
    tools=[search_knowledge_base, add_to_knowledge_base],
    verbose=True,
)

# Simple CLI client
def main():
    print("Welcome to the RAG Agent CLI. Commands: /add <title> <content>, /search <query>, /quit")
    while True:
        try:
            user_input = input(">>> ")
        except EOFError:
            break
        if not user_input.strip():
            continue
        if user_input.startswith("/quit"):
            print("Goodbye!")
            break
        elif user_input.startswith("/add"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: /add <title> <content>")
                continue
            title, content = parts[1], parts[2]
            result = add_to_knowledge_base(content, title)
            print(result)
        elif user_input.startswith("/search"):
            query = user_input[len("/search"):].strip()
            if not query:
                print("Usage: /search <query>")
                continue
            results = search_knowledge_base(query, max_results=3)
            for i, res in enumerate(results, 1):
                print(f"Result {i}: (score={res['score']:.4f}) Title: {res['metadata'].get('title', 'N/A')}")
                print(res["content"][:200] + "...")
        else:
            # Treat as normal user message to agent
            response = agent_executor.invoke({"input": user_input})
            print(response.get("output", ""))

if __name__ == "__main__":
    main()
