from typing import Literal
from langgraph.graph import START, END, MessageGraph
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from memory import memory
from nodes.info import chain
from nodes.prompt import prompt_generation_chain


def get_state(messages) -> Literal["add_tool_message", "info", "__end__"]:
    # if it is an AI message and has a tool call, it is a tool message
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    # if it is not a human message, and doesn't contain a tool call, end.
    elif not isinstance(messages[-1], HumanMessage):
        return END

    return "info"


workflow = MessageGraph()

# nodes
workflow.add_node("info", chain)


@workflow.add_node
def add_tool_message(state: list):
    return ToolMessage(
        content="Prompt generated!",
        tool_call_id=state[-1].tool_calls[0]["id"],
    )


workflow.add_node("prompt", prompt_generation_chain)

# edges
workflow.add_edge(START, "info")
workflow.add_conditional_edges("info", get_state)
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)


graph = workflow.compile(checkpointer=memory)
