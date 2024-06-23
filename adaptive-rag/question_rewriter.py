from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from textwrap import dedent


llm = ChatOllama(model="mistral:7b-instruct-q8_0", temperature=0.0)

re_writer_prompt = PromptTemplate(
    template=dedent("""
        You are a question rewriter who converts an input question to a better version optimized for vectorstore retrieval. Look at the original question and formulate an improved question.
        Here is the original question: {question}
        Improved question with no preamble.
    """),
    input_variables=["question"],
)

# question = "agent memory"
question_rewriter = re_writer_prompt | llm | StrOutputParser()
# print(question_rewriter.invoke({"question": question}))
