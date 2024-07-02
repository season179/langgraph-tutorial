import streamlit as st
from dotenv import load_dotenv, find_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic


load_dotenv(find_dotenv())


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("Part 1: Build a Basic Chatbot with Streamlit and LangGraph")


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
        
    with st.chat_message("Assistant"):
        st.write_stream(graph.stream({"messages": ("user", user_input)}, stream_mode="debug"))
else:
    st.write("No input provided")
