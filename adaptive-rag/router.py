# Router
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from embed import embedding_function
from textwrap import dedent
from langchain_community.chat_models import ChatOllama


prompt = PromptTemplate(
    # template=dedent("""
    #     You are an expert at routing a user question to a vectorstore or web search.
    #     Use the vectorstore for questions on LangGraph.
    #     You can be relaxed with the keywords in the question related to these topics.
    #     For all other topics, you can rely on the simplicity and effectiveness of web search. Based on the question, you can confidently provide a binary choice of 'web_search' or 'vectorstore'.
    #     Return the JSON with a single key 'datasource' and no preamble or explanation.
    #     Question to route: {question}
    # """),
    template=dedent("""
        You are an expert at routing a user question to a vectorstore or web search.
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks.
        You do not need to be stringent with the keywords in the question related to these topics.
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
        Return the a JSON with a single key 'datasource' and no premable or explanation.
        Question to route: {question}
    """),
    input_variables=["question"],
)


llm = ChatOllama(model="mistral:7b-instruct-q8_0", format="json", temperature=0.0)

question_router = prompt | llm | JsonOutputParser()
# question = "langgraph checkpoint."
# db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function, collection_name="langgraph-reference")
# docs = db.similarity_search(question)
# doc_text = docs[1].page_content
# print(question_router.invoke({"question": question}))
