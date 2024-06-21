from typing import List

from textwrap import dedent
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from llm import llm

template = dedent("""
    Your crucial role is to gather information from a user about the type of prompt template they want to create. 
    Your understanding and communication skills are key to this process.

    You should get the following information from them:

    - What is the objective of the prompt is
    - What variables will be passed into the prompt template
    - Any requirements that the output MUST adhere to

    If you are unable to discern this information, ask them to clarify! Do not attempt to guess wildly.

    After you have discerned all the information, call the relevant tool.
""")


def get_mesages_info(messages):
    return [SystemMessage(content=template)] + messages


class PromptInstructions(BaseModel):
    """
    This class contains the instructions on how to prompt the LLM.
    """

    objective: str
    variables: List[str]
    requirements: List[str]


llm_with_tool = llm.bind_tools([PromptInstructions])

chain = get_mesages_info | llm_with_tool
