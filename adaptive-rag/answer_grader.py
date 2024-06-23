from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from textwrap import dedent

llm = ChatOllama(model="mistral:7b-instruct-q8_0", format="json", temperature=0.0)

prompt = PromptTemplate(
    template=dedent("""
        You are a grader assessing whether an answer is helpful to resolve a question.
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score of 'yes' or 'no' to indicate whether the answer is helpful to resolve a question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    """),
    input_variables=["generation", "question"],
)

# generation = dedent("""
#     The document you provided is a blog post about LLM-powered autonomous agents. The author discusses the concept of using large language models (LLMs) as the core controller for autonomous agents, and provides an overview of the different components that make up such a system.

#     One of the key components discussed in the post is memory. The author explains that there are several types of memory in human brains, including sensory memory, short-term memory (STM), and long-term memory (LTM). They also discuss how LLMs can be used to simulate these different types of memory, using techniques such as learning embeddings representations for raw inputs and maximum inner product search (MIPS) to retrieve information from an external vector store.

#     The post also touches on some of the challenges that arise when building LLM-powered autonomous agents, including the finite context length of Transformer, difficulties in long-term planning and task decomposition, and the reliability of natural language interfaces. The author concludes by emphasizing the potential of LLMs as a powerful general problem solver, and encourages readers to explore this exciting area of research further.
# """)

# question = "agent memory"
answer_grader = prompt | llm | JsonOutputParser()
# print(answer_grader.invoke({"question": question, "generation": generation}))
