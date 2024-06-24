import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from nanoid import generate
from textwrap import dedent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

system_prompt = dedent("""
    You are an advanced AI assistant with access to real-time internet search capabilities. Your primary function is to assist users by providing up-to-date and accurate information from the web. Follow these guidelines:

    1. Use your internet search tool whenever you need to find current information or verify facts.
    2. Always cite your sources by providing the website name and URL after each piece of information you retrieve.
    3. If search results are inconclusive or contradictory, communicate this to the user and explain the discrepancies.
    4. Respect user privacy and do not search for or reveal personal information.
    5. If a search doesn't yield relevant results, inform the user and suggest alternative search terms or approaches.
    6. Summarize lengthy search results concisely, highlighting the most relevant points.
    7. When appropriate, offer to perform follow-up searches to gather more detailed information.
    8. If a user's question is unclear, ask for clarification before conducting a search.
    9. Be objective and present multiple viewpoints on controversial topics.
    10. Inform users of the date of the information you find, especially for time-sensitive queries.
    11. If you encounter any errors or limitations with the search tool, communicate this clearly to the user.

    Remember, your goal is to be a helpful, accurate, and efficient research assistant, leveraging the power of internet search to provide the best possible assistance to users.
""")

@st.cache_resource
def create_agent():
    memory = SqliteSaver.from_conn_string(":memory:")
    model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
    search = TavilySearchResults(max_results=2)
    tools = [search]
    return create_react_agent(model, tools, checkpointer=memory, messages_modifier=system_prompt)

st.title("AI Assistant with Internet Search")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

agent_executor = create_agent()

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=prompt)]},
            {"configurable": {"thread_id": st.session_state.thread_id}},
            stream_mode="values"
        ):
            last_message = chunk["messages"][-1]

            if not isinstance(last_message, HumanMessage):
                if isinstance(last_message.content, str):
                    full_response += last_message.content
                    message_placeholder.markdown(full_response + "▌")
                else:
                    for item in last_message.content:
                        if item["type"] == "text":
                            full_response += f"*AI thought:* {item['text']}\n\n"
                        elif item["type"] == "tool_use":
                            full_response += f"*AI used {item['name']} with input:* {item['input']['query']}\n\n"
                        else:
                            full_response += f"*Unhandled:* {item}\n\n"
                        message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.title("About")
st.sidebar.info("This is an AI assistant with internet search capabilities. Ask any question, and it will provide up-to-date information from the web.")
