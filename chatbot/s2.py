import asyncio
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
async def run_chatbot():
    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    state = {"messages": [], "ask_human": False}

    while True:
        user_input = input("Human: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        async for event in graph.astream(state, config):
            if "chatbot" in event:
                ai_message = event["chatbot"]["messages"][-1]
                if isinstance(ai_message, AIMessage):
                    print("AI:", ai_message.content)
            elif "tools" in event:
                kind = event["tools"]["event"]
                if kind == "on_tool_start":
                    print("--")
                    print(
                        f"Starting tool: {event['tools']['name']} with inputs: {event['tools']['data'].get('input')}"
                    )
                elif kind == "on_tool_end":
                    print(f"Done tool: {event['tools']['name']}")
                    print(f"Tool output was: {event['tools']['data'].get('output')}")
                    print("--")


if __name__ == "__main__":
    asyncio.run(run_chatbot())
