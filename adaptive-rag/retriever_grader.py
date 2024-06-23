from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from embed import embedding_function
from textwrap import dedent
# from llm import llm
from langchain_community.chat_models import ChatOllama


prompt = PromptTemplate(
    template=dedent("""
        You are a grader assessing the relevance of a retrieved document to a user question. 
        Here is the retrieved document: {document}
        Here is the user question: {question}
        If the document contains keywords related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Remember, the scoring system is binary. You simply need to give a 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    """),
    input_variables=["question", "document"],
)

llm = ChatOllama(model="mistral:7b-instruct-q8_0", format="json", temperature=0.0)

retrieval_grader = prompt | llm | JsonOutputParser()

# question = "langgraph checkpoint"
# db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function, collection_name="langgraph-reference")
# docs = db.similarity_search(question)
# doc_text = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_text}))
