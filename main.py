import os
from typing import List, Dict

# Современные импорты LangChain и Ollama
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

# 1. Инициализация эмбеддингов
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Локальный клиент Qdrant (в памяти для тестирования, либо укажите url/path)
client = QdrantClient(location=":memory:")
collection_name = "knowledge_base"

# Создаем векторное хранилище
# Важно: в langchain_qdrant метод from_documents или явный конструктор требует qdrant_client
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

# 3. Настройка чанкера
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


# 4. Определение RAG-инструментов через @tool
@tool
def search_knowledge_base(query: str, max_results: int = 3) -> str:
    """Семантический поиск в базе знаний. Возвращает текстовую строку с найденными кусками данных."""
    # similarity_search_with_relevance_scores возвращает кортежи (Document, score)
    results = vector_store.similarity_search_with_relevance_scores(query, k=max_results)
    
    if not results:
        return "В базе знаний ничего не найдено по этому запросу."
        
    formatted_results = []
    for idx, (doc, score) in enumerate(results, 1):
        title = doc.metadata.get("title", "Без названия")
        formatted_results.append(
            f"Документ {idx} [{title}] (Релевантность: {score:.2f}):\n{doc.page_content}"
        )
    return "\n\n".join(formatted_results)


@tool
def add_to_knowledge_base(content: str, title: str = "Инструкция") -> str:
    """Добавление нового документа или текста в базу знаний с автоматическим разбиением на чанки."""
    chunks = splitter.split_text(content)
    docs: List[Document] = []
    for i, chunk in enumerate(chunks):
        meta = {"title": title, "chunk_index": i}
        docs.append(Document(page_content=chunk, metadata=meta))
    
    vector_store.add_documents(docs)
    return f"Успешно добавлено. Документ '{title}' разбит на {len(chunks)} чанков и сохранен."


# Список инструментов для агента
tools = [search_knowledge_base, add_to_knowledge_base]

# 5. Инициализация современной LLM с поддержкой вызова инструментов
llm = ChatOllama(
    model="llama3", 
    temperature=0,
    # Некоторые версии Ollama требуют явного включения, если модель поддерживает native tools
)

# 6. Системный промпт для RAG-агента
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Ты — полезный AI-агент с доступом к локальной базе знаний (RAG).\n"
        "Твоя задача — отвечать на вопросы пользователя. Если у тебя нет ответа, "
        "обязательно используй инструмент `search_knowledge_base` для поиска информации.\n"
        "Если пользователь делится важными фактами, которых нет в твоей памяти, "
        "используй `add_to_knowledge_base`, чтобы сохранить их.\n"
        "Всегда отвечай на русском языке."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Создание современного агента (взамен устаревшего OpenAI functions)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# 7. Интерактивный CLI-клиент
if __name__ == "__main__":
    print("="*60)
    print("Добро пожаловать в RAG-Agent CLI!")
    print("Команды:\n  /add <название> | <текст> - Добавить в базу напрямую")
    print("  /search <запрос>           - Прямой поиск по базе")
    print("  /quit                      - Выход")
    print("  Любой другой текст расценивается как обращение к Агенту.")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nВы > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
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
                print("Ошибка формата! Используйте: /add Название документа | Текст документа")
                continue
            
            # Вызов функции напрямую (минуя агента)
            result = add_to_knowledge_base.invoke({"content": content, "title": title})
            print(f"[Система]: {result}")
            
        elif user_input.startswith("/search"):
            try:
                _, query = user_input.split(" ", 1)
                query = query.strip()
            except ValueError:
                print("Ошибка формата! Используйте: /search Ваш поисковый запрос")
                continue
                
            # Вызов поиска напрямую
            result = search_knowledge_base.invoke({"query": query})
            print(f"[База Знаний]:\n{result}")
            
        else:
            # Обращение к LLM-агенту, который сам решит, вызывать ли инструменты
            print("[Агент думает...]")
            try:
                response = agent_executor.invoke({"input": user_input})
                print(f"Ответ Агента: {response['output']}")
            except Exception as e:
                print(f"Произошла ошибка при работе агента: {e}")