from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)
memory = SqliteSaver.from_conn_string("./checkpoints.sqlite")
config = {"configurable": {"thread_id": "1"}}


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
graph = graph_builder.compile(checkpointer=memory)


# Add this function after the graph definition
def run_chatbot():
    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    state = {"messages": [], "ask_human": False}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        for response in graph.stream(state, config):
            if response["chatbot"]:
                ai_message = response["chatbot"]["messages"][-1]
                if isinstance(ai_message, AIMessage):
                    print("AI:", ai_message.content)


if __name__ == "__main__":
    run_chatbot()
