import json
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv(find_dotenv())


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
tool = TavilySearchResults(max_results=2)
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools([tool])


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


class BasicToolNode:
    """
    A node that runs the tools requested in the last AIMessage.
    """

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages in inputs")

        outputs = []

        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}


def route_tools(state: State) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    return "__end__"


tool_node = BasicToolNode(tools=[tool])

graph_builder.set_entry_point("chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", "__end__": "__end__"},
)
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("Part 2: Enhancing the Chatbot with Tools ")


for message in st.session_state.messages:
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            with st.chat_message("User"):
                st.write(content)
        elif role == "assistant":
            with st.chat_message("Assistant"):
                st.write(content)


user_input = st.chat_input("Enter your message")

if user_input and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("User"):
        st.write(user_input)

    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print(value)

            if isinstance(value["messages"][-1], BaseMessage):
                last_message = value["messages"][-1].content

                with st.chat_message("Assistant"):
                    st.write(last_message)

                st.session_state.messages.append(
                    {"role": "assistant", "content": last_message}
                )
else:
    st.write("No input provided")
