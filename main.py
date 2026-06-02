"""
Agent with RAG memory using ChromaDB instead of Qdrant.
"""
import os
from pathlib import Path
from typing import List, Dict

# LangChain imports
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.llms import Ollama

# Configuration
CHROMA_DIR = "./chroma"
DOCS_DIR = "./docs"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Initialize embeddings and LLM
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
lm = Ollama(model=LLM_MODEL, temperature=0.2)

# Initialize or load Chroma vector store
vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

@tool("search_knowledge_base")
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Semantic search in the knowledge base."""
    results = vector_store.similarity_search_with_score(query, k=max_results)
    return [{"content": doc.page_content, "score": score} for doc, score in results]

@tool("add_to_knowledge_base")
def add_to_knowledge_base(content: str, title: str) -> str:
    """Add a document to the knowledge base."""
    docs = text_splitter.split_text(content)
    vector_store.add_documents([{"page_content": d, "metadata": {"title": title}} for d in docs])
    return f"Document '{title}' added with {len(docs)} chunks."

# Agent setup
tools = [search_knowledge_base, add_to_knowledge_base]
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=create_openai_functions_agent(llm=lm, tools=tools),
    tools=tools,
    verbose=True,
)

def load_documents_from_dir(directory: str):
    """Load all text files from a directory into the vector store."""
    for file_path in Path(directory).glob("**/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        title = file_path.stem
        add_to_knowledge_base(content, title)

if __name__ == "__main__":
    # Load existing docs if any
    load_documents_from_dir(DOCS_DIR)
    print("Agent ready. Use /add <title> <path>, /search <query>, or /quit.")
    while True:
        user_input = input(">>> ").strip()
        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        if user_input.startswith("/add"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: /add <title> <path>")
                continue
            title, path = parts[1], parts[2]
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                result = add_to_knowledge_base(content, title)
                print(result)
            except Exception as e:
                print(f"Error adding document: {e}")
        elif user_input.startswith("/search"):
            query = user_input[len("/search"):].strip()
            if not query:
                print("Usage: /search <query>")
                continue
            results = search_knowledge_base(query, max_results=3)
            for i, res in enumerate(results, 1):
                print(f"{i}. (score {res['score']:.4f}) {res['content'][:200]}...")
        else:
            # Treat as normal query to agent
            response = agent_executor.invoke({"input": user_input})
            print(response["output"])
