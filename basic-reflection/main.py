from typing import List, Sequence
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, MessageGraph


load_dotenv(find_dotenv())


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 3-paragraph essays."
            " Generate the best essay possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", max_tokens=4096)
llm = ChatOllama(model="qwen2:7b-instruct-q8_0")
# llm = ChatOllama(model="llama3:8b-instruct-q8_0")

generate = prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
            " Provide detailed constructive recommendations, including requests for length, depth, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect = reflection_prompt | llm


def generation_node(state: Sequence[BaseMessage]):
    return generate.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}

    # Frist message is the original user request. We hold it the same for all nodes
    translated = [messages[0]] + [
        cls_map[message.type](content=message.content) for message in messages[1:]
    ]

    response = reflect.invoke({"messages": translated})

    return HumanMessage(content=response.content)


builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        # End after 3 iterations
        return END

    return "reflect"


builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()

for event in graph.stream(
    [
        HumanMessage(
            content="Write an essay on how AI agent can help increase a software engineer's productivity."
        ),
    ]
):
    print(event)
    print("---")
