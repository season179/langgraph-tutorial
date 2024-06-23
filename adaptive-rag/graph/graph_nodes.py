import os
from langchain.schema import Document
from langchain_chroma import Chroma
from embed import embedding_function
from generate import rag_chain
from retriever_grader import retrieval_grader
from question_rewriter import question_rewriter
from tavily import TavilyClient
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function,
    collection_name="rag-chroma",
)


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current state of the graph

    Returns:
        state (dict): New key added to state, "documents", that contains retrieved documents.
    """

    print("---RETRIEVE---")
    question = state["question"]

    documents = db.similarity_search(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current state of the graph

    Returns:
        state (dict): New key added to state, "generation", that contains generation.
    """

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "generation": generation, "question": question}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current state of the graph

    Returns:
        state (dict): Updated "documents" key with only filtered relevant documents.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []

    for document in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )

        grade = score["score"]

        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(document)
        else:
            print("---GRADE: DOCUMENT IRRELEVANT---")
            continue

    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current state of the graph

    Returns:
        state (dict): Updated "question" key with a rephrased question.
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Rewrite question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    Search the web based on the question.

    Args:
        state (dict): The current state of the graph

    Returns:
        state (dict): Updated "documents" key with web search results.
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Search web
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    result = tavily.qna_search(query=question, search_depth="advanced")
    print(result)
    web_results = Document(page_content=result)

    return {"documents": web_results, "question": question}
