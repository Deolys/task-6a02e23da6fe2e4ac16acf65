import os
from typing import List, Dict

# Современные импорты LangChain
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

# 1. Настройка эмбеддингов через Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Локальный клиент Qdrant (в памяти для тестирования)
# Для Docker используйте: client = QdrantClient(url="http://localhost:6333")
qdrant_client = QdrantClient(location=":memory:")
collection_name = "knowledge_base"

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embeddings,
)

# 3. Система чанкинга
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


# 4. RAG-инструменты для агента через декоратор @tool
@tool
def search_knowledge_base(query: str, max_results: int = 3) -> str:
    """Семантический поиск в базе знаний. Возвращает найденные текстовые фрагменты."""
    # Используем поиск с метрикой релевантности
    results = vector_store.similarity_search_with_relevance_scores(query, k=max_results)
    
    if not results:
        return "В базе знаний не найдено релевантной информации."
        
    formatted_results = []
    for idx, (doc, score) in enumerate(results, 1):
        title = doc.metadata.get("title", "Без названия")
        formatted_results.append(
            f"Документ {idx} [{title}] (Релевантность: {score:.2f}):\n{doc.page_content}"
        )
    return "\n\n".join(formatted_results)


@tool
def add_to_knowledge_base(content: str, title: str = "Инструкция") -> str:
    """Добавление документа в базу знаний с автоматическим разбиением на чанки."""
    chunks = splitter.split_text(content)
    docs: List[Document] = []
    
    for i, chunk in enumerate(chunks):
        meta = {"title": title, "chunk_index": i}
        docs.append(Document(page_content=chunk, metadata=meta))
        
    vector_store.add_documents(docs)
    return f"Успешно добавлено. Документ '{title}' разбит на {len(chunks)} чанков и сохранен в Qdrant."


# Список инструментов, доступных агенту
tools = [search_knowledge_base, add_to_knowledge_base]

# 5. Инициализация современной модели ChatOllama
llm = ChatOllama(
    model="llama3",
    temperature=0  # Низкая температура для точности работы RAG
)

# Системный промпт, инструктирующий агента использовать базу знаний
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Ты — экспертный AI-агент с интеграцией RAG-памяти.\n"
        "Твоя задача — отвечать на вопросы пользователя на основе доступных знаний.\n"
        "Если у тебя нет точной информации для ответа, ты ОБЯЗАН вызвать инструмент `search_knowledge_base`.\n"
        "Если пользователь предоставляет новые важные факты или инструкции, сохрани их через `add_to_knowledge_base`.\n"
        "Всегда общайся с пользователем на русском языке."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Создание агента и исполнителя (AgentExecutor)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)


# 6. Интерактивный тестовый клиент (CLI)
if __name__ == "__main__":
    print("=" * 60)
    print("Запущен интерактивный клиент RAG-Агента.")
    print("Доступные команды:")
    print("  /add <название> | <текст> - Прямое добавление в базу")
    print("  /search <запрос>           - Прямой семантический поиск")
    print("  /quit                      - Выход из программы")
    print("  Любой другой текст будет передан Агенту для анализа.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nВы > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            break
            
        if not user_input:
            continue
            
        if user_input.startswith("/quit"):
            print("До свидания!")
            break
            
        elif user_input.startswith("/add"):
            try:
                _, rest = user_input.split(" ", 1)
                title, content = rest.split("|", 1)
                title = title.strip()
                content = content.strip()
            except ValueError:
                print("Ошибка! Формат: /add Название | Текст документа")
                continue
                
            # Вызов логики инструмента напрямую для обхода агента
            res = add_to_knowledge_base.invoke({"content": content, "title": title})
            print(f"[Система]: {res}")
            
        elif user_input.startswith("/search"):
            try:
                _, query = user_input.split(" ", 1)
                query = query.strip()
            except ValueError:
                print("Ошибка! Формат: /search Ваш запрос")
                continue
                
            res = search_knowledge_base.invoke({"query": query})
            print(f"[База Данных]:\n{res}")
            
        else:
            # Отправка свободного запроса агенту
            print("[Агент обрабатывает запрос...]")
            try:
                response = agent_executor.invoke({"input": user_input})
                print(f"\nОтвет Агента:\n{response['output']}")
            except Exception as e:
                print(f"Ошибка выполнения: {e}")