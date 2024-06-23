import streamlit as st
from nanoid import generate
from workflow import graph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv

unique_id = generate()
load_dotenv(find_dotenv())
config = {"configurable": {"thread_id": str(unique_id)}}


def main():
    st.title("Prompt Generator")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(generate())
        st.session_state.messages = []

    user_input = st.text_input("User input:", key="user_input")

    if st.button("Generate"):
        if user_input:
            st.session_state.messages.append({"message": user_input, "is_user": True})

            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Create a placeholder for the output
            output_placeholder = st.empty()

            for output in graph.stream(
                [HumanMessage(content=user_input)],
                config=config,
                stream_mode="updates",
            ):
                last_message = next(iter(output.values()))
                # Update the placeholder with the new content
                output_placeholder.write(last_message.content)

            if output and "prompt" in output:
                st.success("Done!")
                st.session_state.messages.append(
                    {"message": last_message.content, "is_user": False}
                )

    st.subheader("Chat History")
    for msg in st.session_state.messages:
        if msg["is_user"]:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(msg["message"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["message"])


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()

# while True:
#     user = input("User (q/Q to quit): ")

#     if user in {"q", "Q"}:
#         break

#     output = None

#     for output in graph.stream(
#         [HumanMessage(content=user)],
#         config=config,
#         stream_mode="updates",
#     ):
#         last_message = next(iter(output.values()))
#         last_message.pretty_print()

#     if output and "prompt" in output:
#         print("Done!")
