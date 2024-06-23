from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from textwrap import dedent
from embed import embedding_function
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOllama


load_dotenv(find_dotenv())


prompt = PromptTemplate(
    template=dedent("""
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved-context to answer the question. 
        If you don't know the answer, say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer: 
    """),
    input_variables=["question", "context"],
)


# post-processing
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# llm = ChatOllama(model="llama3:8b-instruct-q8_0", temperature=0.0)
# llm = ChatOllama(model="qwen2:7b-instruct-q8_0", temperature=0.0)
llm = ChatOllama(model="mistral:7b-instruct-q8_0", temperature=0.0)


# chain
rag_chain = prompt | llm | StrOutputParser()

# run
# question = "agent memory"
# db = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embedding_function,
#     collection_name="rag-chroma",
# )

# docs = db.similarity_search(question)
# print(f"Docs length: {len(docs)}")
# print(docs[0].page_content)
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)
