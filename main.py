"""
Main implementation of an RAG agent using Qdrant, Ollama and LangChain.
The code follows the instructor feedback:
- Uses Qdrant as vector store instead of ChromaDB.
- Uses current LangChain APIs (v0.2).
- Embeddings are generated with Ollama (`nomic-embed-text`).
- LLM is Ollama (`llama3`).
"""

import os
from pathlib import Path
from typing import List, Dict

# Load environment variables if any
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

# LangChain imports (v0.2)
from langchain_community.vectorstores.qdrant import QdrantVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Configuration
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_agent_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# -----------------------------
# Vector store helper
# -----------------------------
class QdrantStore:
    def __init__(self, url: str = QDRANT_URL, collection_name: str = QDRANT_COLLECTION):
        self.url = url
        self.collection_name = collection_name
        # Initialize embeddings and vector store lazily
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.store: QdrantVectorStore | None = None

    def _ensure_store(self):
        if self.store is None:
            self.store = QdrantVectorStore.from_texts(
                texts=[],  # start empty; will add later
                embedding=self.embeddings,
                url=self.url,
                collection_name=self.collection_name,
            )

    def add_documents(self, docs: List[Document]):
        self._ensure_store()
        self.store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        self._ensure_store()
        return self.store.similarity_search(query=query, k=k)

# -----------------------------
# Tools for the agent
# -----------------------------
store = QdrantStore()

def _search_knowledge_base(query: str, max_results: int = 4) -> List[Dict]:
    """Return top documents as list of dicts with content and metadata."""
    docs = store.similarity_search(query=query, k=max_results)
    return [{"content": d.page_content, **d.metadata} for d in docs]

def _add_to_knowledge_base(content: str, title: str) -> str:
    """Add a single document to the vector store."""
    doc = Document(page_content=content, metadata={"title": title})
    store.add_documents([doc])
    return f"Document '{title}' added successfully."

search_tool = Tool(
    name="search_knowledge_base",
    func=_search_knowledge_base,
    description=(
        "Search the knowledge base for relevant documents.\n"
        "Parameters:\n"
        "- query (str): The search query.\n"
        "- max_results (int, optional): Number of results to return. Defaults to 4."
    ),
)

add_tool = Tool(
    name="add_to_knowledge_base",
    func=_add_to_knowledge_base,
    description=(
        "Add a new document to the knowledge base.\n"
        "Parameters:\n"
        "- content (str): The full text of the document.\n"
        "- title (str): A short title for the document."
    ),
)

# -----------------------------
# Agent setup
# -----------------------------
llm = Ollama(model=LLM_MODEL, temperature=0.7)
agent = create_openai_functions_agent(llm=llm, tools=[search_tool, add_tool])
executor = AgentExecutor(agent=agent, tools=[search_tool, add_tool], verbose=True)

# -----------------------------
# Document ingestion helper (used by init script)
# -----------------------------
def ingest_directory(directory: str):
    """Load all .txt files from a directory, chunk them and add to store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs: List[Document] = []
    for path in Path(directory).rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            meta = {"source": str(path), "chunk_index": i}
            docs.append(Document(page_content=chunk, metadata=meta))
    store.add_documents(docs)

# -----------------------------
# Interactive CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init command
    init_parser = subparsers.add_parser("init", help="Load documents from a directory into the vector store")
    init_parser.add_argument("dir", type=str, help="Directory containing .txt files to ingest")

    # add command
    add_parser = subparsers.add_parser("add", help="Add a document via CLI")
    add_parser.add_argument("title", type=str, help="Title of the document")
    add_parser.add_argument("file", type=str, help="Path to text file containing content")

    # search command
    search_parser = subparsers.add_parser("search", help="Search knowledge base")
    search_parser.add_argument("query", type=str, help="Query string")
    search_parser.add_argument("-k", type=int, default=4, help="Number of results")

    # chat command (interactive agent)
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat with the agent")

    args = parser.parse_args()

    if args.command == "init":
        ingest_directory(args.dir)
        print(f"Ingested documents from {args.dir}")
    elif args.command == "add":
        content = Path(args.file).read_text(encoding="utf-8")
        result = _add_to_knowledge_base(content, args.title)
        print(result)
    elif args.command == "search":
        results = _search_knowledge_base(args.query, max_results=args.k)
        for i, res in enumerate(results, 1):
            print(f"Result {i}: {res.get('title', 'no title')}\n{res['content'][:500]}...\n")
    elif args.command == "chat":
        print("Enter your messages. Type /quit to exit.")
        while True:
            user_input = input("You: ")
            if user_input.strip() == "/quit":
                break
            response = executor.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
    else:
        parser.print_help()
"