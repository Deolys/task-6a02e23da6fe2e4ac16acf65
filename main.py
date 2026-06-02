"""
RAG Agent with Qdrant and Ollama
Author: Auto-generated for task 6a02e23da6fe2e4ac16acf65
"""
import os
from pathlib import Path
from typing import List, Dict

# LangChain imports
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import Document

# -----------------------------
# Configuration
# -----------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:latest"

# -----------------------------
# Vector store setup
# -----------------------------
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = QdrantVectorStore(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

# -----------------------------
# Text splitter
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# -----------------------------
# Tools
# -----------------------------
@tool("search_knowledge_base", "Search the knowledge base for relevant documents.")
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Return a list of dictionaries with keys 'title' and 'content'."""
    docs = vector_store.similarity_search_with_score(query, k=max_results)
    results = []
    for doc, score in docs:
        meta = doc.metadata or {}
        results.append({
            "title": meta.get("title", "No title"),
            "content": doc.page_content,
            "score": f"{score:.4f}",
        })
    return results

@tool("add_to_knowledge_base", "Add a new document to the knowledge base.")
def add_to_knowledge_base(content: str, title: str = "Untitled") -> str:
    """Split content into chunks and store them with metadata."""
    chunks = text_splitter.split_text(content)
    metadatas = [{"title": title} for _ in chunks]
    vector_store.add_texts(chunks, metadatas=metadatas)
    return f"Added {len(chunks)} chunk(s) to the knowledge base under title '{title}'."

# -----------------------------
# Agent setup
# -----------------------------
TOOLS = [search_knowledge_base, add_to_knowledge_base]
AGENT = create_openai_functions_agent(LLM_MODEL, TOOLS)
EXECUTOR = AgentExecutor(agent=AGENT, tools=TOOLS, verbose=True)

# -----------------------------
# CLI
# -----------------------------
def print_help():
    help_text = """
Commands:
  /add <title> | <content>   Add a document to the knowledge base.
  /search <query>            Search the knowledge base.
  /quit                      Exit the program.
  /help                      Show this help message.
"""
    print(help_text)

def main():
    print("RAG Agent CLI. Type /help for commands.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.startswith("/add"):
            parts = user_input.split("|", 1)
            if len(parts) != 2:
                print("Usage: /add <title> | <content>")
                continue
            title, content = [p.strip() for p in parts]
            result = add_to_knowledge_base(content=content, title=title)
            print(result)
        elif user_input.startswith("/search"):
            query = user_input[len("/search"):].strip()
            if not query:
                print("Please provide a search query.")
                continue
            results = search_knowledge_base(query=query, max_results=5)
            if not results:
                print("No relevant documents found.")
            else:
                for i, res in enumerate(results, 1):
                    print(f"\nResult {i}: (score {res['score']}) Title: {res['title']}\n{res['content'][:500]}...")
        elif user_input == "/quit":
            print("Goodbye!")
            break
        elif user_input == "/help":
            print_help()
        else:
            # Treat as a normal prompt to the agent
            response = EXECUTOR.invoke({"input": user_input})
            print(response.get("output", ""))

if __name__ == "__main__":
    main()
