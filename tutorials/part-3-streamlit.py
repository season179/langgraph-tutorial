import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# Can't use in-memory db as st restarts it every time
memory = SqliteSaver.from_conn_string("part-3.sqlite")
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
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

st.title("Part 3: Chatbot with memory")

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

    for event in graph.stream(
        {"messages": ("user", user_input)}, config, stream_mode="updates"
    ):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                last_message = value["messages"][-1].content

                with st.chat_message("Assistant"):
                    st.write(last_message)

                st.session_state.messages.append(
                    {"role": "assistant", "content": last_message}
                )
                
        print(graph.get_state(config))
