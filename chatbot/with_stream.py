import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())


# Set up the model
@st.cache_resource
def get_model():
    return ChatAnthropic(model="claude-3-sonnet-20240229")


model = get_model()

# Set up the Streamlit app
st.title("Simple Chatbot with Claude")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to chat about?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in model.stream([HumanMessage(content=prompt)]):
            full_response += chunk.content
            message_placeholder.write(full_response + "▌")

        message_placeholder.write(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display information about the app
st.sidebar.title("About")
st.sidebar.info(
    "This is a simple chatbot using Claude 3 Sonnet. It demonstrates streaming capabilities in Streamlit."
)
