# main.py
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_agent, Tool, AgentExecutor
from langchain.schema import HumanMessage
import json

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Initialize embeddings and LLM
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
lm = OllamaLLM(model=LLM_MODEL, temperature=0.7)

# Ensure Qdrant collection exists
vector_store = QdrantVectorStore(
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

# Text splitter for chunking documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Tool: search knowledge base
@Tool(name="search_knowledge_base", description="Search the knowledge base for relevant information.")
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    results = vector_store.similarity_search_with_score(query, k=max_results)
    docs = [r[0].page_content for r in results]
    return json.dumps(docs)

# Tool: add to knowledge base
@Tool(name="add_to_knowledge_base", description="Add a document to the knowledge base.")
def add_to_knowledge_base(content: str, title: str = "") -> str:
    chunks = splitter.split_text(content)
    metadatas = [{"title": title} for _ in chunks]
    vector_store.add_texts(chunks, metadatas=metadatas)
    return f"Added {len(chunks)} chunks to the knowledge base."

# Agent setup
tools = [search_knowledge_base, add_to_knowledge_base]
agent_executor = create_agent(
    llm=lm,
    tools=tools,
    system_prompt="You are an assistant that can search and add documents to a knowledge base.",
)

# CLI loop
if __name__ == "__main__":
    print("Welcome to the RAG agent. Commands: /add <title> <content>, /search <query>, /quit")
    while True:
        user_input = input(">>> ")
        if not user_input:
            continue
        if user_input.startswith("/quit"):
            print("Goodbye!")
            break
        elif user_input.startswith("/add"):
            try:
                _, title, content = user_input.split(maxsplit=2)
                response = add_to_knowledge_base(content, title)
                print(response)
            except ValueError:
                print("Usage: /add <title> <content>")
        elif user_input.startswith("/search"):
            try:
                _, query = user_input.split(maxsplit=1)
                response = search_knowledge_base(query)
                docs = json.loads(response)
                for i, doc in enumerate(docs, 1):
                    print(f"Result {i}:\n{doc}\n")
            except ValueError:
                print("Usage: /search <query>")
        else:
            # Treat as normal query to agent
            result = agent_executor.invoke(HumanMessage(content=user_input))
            print(result)
