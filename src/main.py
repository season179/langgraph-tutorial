from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv, find_dotenv

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv(find_dotenv())


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-haiku-20240307")
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

memory = SqliteSaver.from_conn_string(":memory:")

graph = graph_builder.compile(checkpointer=memory, interrupt_after=["tools"])

config = { "configurable": { "thread_id": "1" } }

while True:
    user_input = input("User: ")

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    events = graph.stream(
        { "messages": [("user", user_input)] },
        config=config,
        stream_mode="values"
    )

    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
            
            snapshot = graph.get_state(config)
            psnapshot_next = snapshot.next
            print(f"Snapshot: {psnapshot_next}")
            
            
            if "messages" in snapshot.values and snapshot.values["messages"]:
                existing_message = snapshot.values["messages"][-1]
                print(f"existing message: {existing_message}")
                if hasattr(existing_message, 'tool_calls'):
                    existing_message.tool_calls
        
        
        
        # for value in event.values():
        #     if isinstance(value["messages"][-1], BaseMessage):
        #         print("Assistant:", value["messages"][-1].content)
