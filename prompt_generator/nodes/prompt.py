from textwrap import dedent
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from llm import llm

system_prompt = dedent("""
    You are a superintelligent AI prompt generator. Thoroughly understand the given requirements and act accordingly.
""")

human_prompt = dedent("""
    Based on the following requirements, write a good prompt template:

    {reqs}
""")


# Function to get the messages for the prompt
# Will only get the messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_messages = []

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call = message.tool_calls[0]["args"]
        elif isinstance(message, ToolMessage):
            continue
        elif tool_call is not None:
            other_messages.append(message)

    response = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt.format(reqs=tool_call)),
    ] + other_messages

    return response


prompt_generation_chain = get_prompt_messages | llm
