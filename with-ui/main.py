import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.render import format_tool_to_openai_function
from langgraph.prebuilt import ToolExecutor, chat_agent_executor

# Define tools
tools = [TavilySearchResults(max_results=1)]
tool_executor = ToolExecutor(tools)

# Define model
model = ChatOpenAI(temperature=0, streaming=True)

# Bind functions to model
functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)

# Create the agent executor
app = chat_agent_executor.create_function_calling_executor(model, tools)


# Streamlit app definition
def main():
    st.title("LangGraph and Streamlit Integration")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Say something"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the input with LangGraph agent
        inputs = {"messages": [HumanMessage(content=prompt)]}
        response_container = st.empty()

        for output in app.stream(inputs):
            for key, value in output.items():
                assistant_message = value["messages"][0].content
                with st.chat_message("assistant"):
                    st.markdown(assistant_message)
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_message}
                )


if __name__ == "__main__":
    main()
