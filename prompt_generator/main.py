from nanoid import generate
from workflow import graph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv

unique_id = generate()
load_dotenv(find_dotenv())
config = {"configurable": {"thread_id": str(unique_id)}}


while True:
    user = input("User (q/Q to quit): ")

    if user in {"q", "Q"}:
        break

    output = None

    for output in graph.stream(
        [HumanMessage(content=user)],
        config=config,
        stream_mode="updates",
    ):
        last_message = next(iter(output.values()))
        last_message.pretty_print()

    if output and "prompt" in output:
        print("Done!")
