import getpass
import os
from dotenv import load_dotenv, find_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic


load_dotenv(find_dotenv())
llm = ChatAnthropic(model="claude-3-haiku-20240307")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    

graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")

graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye")
        break
    
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)
