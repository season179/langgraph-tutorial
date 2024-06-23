from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from embed import embedding_function
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


urls = [
    # "https://langchain-ai.github.io/langgraph/reference/graphs/",
    # "https://langchain-ai.github.io/langgraph/reference/checkpoints/",
    # "https://langchain-ai.github.io/langgraph/reference/prebuilt/",
    # "https://langchain-ai.github.io/langgraph/reference/errors/"
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
# Flatten the list of lists into a single list of documents
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)


# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    # collection_name="langgraph-reference",
    collection_name="rag-chroma",
    embedding=embedding_function,
    persist_directory="./chroma_db",
)

query = "Tell me about LangGraph checkpoint."
docs = vectorstore.similarity_search(query)

print(docs[0].page_content)