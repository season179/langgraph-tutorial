from typing import Annotated
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

memory = SqliteSaver.from_conn_string("part-4.sqlite")


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt __after__ actions, if desired.
    # interrupt_after=["tools"]
)

config = {"configurable": {"thread_id": "1"}}

st.title("Part 4: Human in the loop")

if "messages" not in st.session_state:
    st.session_state.messages = []

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

    input = {"messages": [("user", user_input)]}

    graph_state = graph.get_state(config)
    print(f"graph state: {graph_state}")

    if (
        user_input == "proceed"
        and hasattr(graph_state, "next")
        and len(graph_state.next) > 0
    ):
        input = None

    for event in graph.stream(input, config, stream_mode="values"):
        # print(event.values())
        for value in event.values():
            print(value)
            if isinstance(value[-1], AIMessage):
                last_message = value[-1].content

                with st.chat_message("Assistant"):
                    st.write(last_message)

                st.session_state.messages.append(
                    {"role": "assistant", "content": last_message}
                )
