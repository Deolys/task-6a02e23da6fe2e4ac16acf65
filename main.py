import os
from typing import List, Dict

# LangChain imports
from langchain_community.vectorstores.qdrant import QdrantVectorStore
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.llms import Ollama

# Initialize embeddings and vector store
embedding = OllamaEmbeddings(model="nomic-embed-text")
qdrant_client = QdrantVectorStore(
    client=None,
    collection_name="knowledge_base",
    embedding=embedding,
)

# Text splitter from new package
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

@tool("search_knowledge_base")
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """Semantic search in the knowledge base."""
    results = qdrant_client.similarity_search_with_score(query, k=max_results)
    return [{"content": r.page_content, "score": r.metadata.get("score", None)} for r in results]

@tool("add_to_knowledge_base")
def add_to_knowledge_base(content: str, title: str = "") -> str:
    """Add a document to the knowledge base."""
    chunks = splitter.split_text(content)
    docs: List[Document] = []
    for i, chunk in enumerate(chunks):
        meta = {"title": title, "chunk_index": i}
        docs.append(Document(page_content=chunk, metadata=meta))
    qdrant_client.add_documents(docs)
    return f"Added {len(chunks)} chunks to the knowledge base."

# LLM and agent setup
llm = Ollama(model="llama3")
agent_executor = AgentExecutor.from_agent_and_tools(
    create_openai_functions_agent(llm, [search_knowledge_base, add_to_knowledge_base]),
    tools=[search_knowledge_base, add_to_knowledge_base],
    verbose=True,
)

# Simple CLI for demonstration
if __name__ == "__main__":
    print("Welcome to the RAG Agent CLI. Commands: /add, /search, /quit")
    while True:
        try:
            user_input = input("> ")
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.startswith("/quit"):
            print("Goodbye!")
            break
        elif user_input.startswith("/add"):
            # Expect format: /add title | content
            try:
                _, rest = user_input.split(" ", 1)
                title, content = rest.split("|", 1)
                title = title.strip()
                content = content.strip()
            except ValueError:
                print("Usage: /add <title> | <content>")
                continue
            result = add_to_knowledge_base(content, title)
            print(result)
        elif user_input.startswith("/search"):
            try:
                _, query = user_input.split(" ", 1)
            except ValueError:
                print("Usage: /search <query>")
                continue
            results = search_knowledge_base(query, max_results=3)
            for idx, res in enumerate(results, 1):
                print(f"Result {idx} (score={res.get('score')}):\n{res['content']}\n")
        else:
            # Treat as agent prompt
            response = agent_executor.invoke({"input": user_input})
            print(response)
