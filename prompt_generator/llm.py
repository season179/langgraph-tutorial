# import os
from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
anthropic_model = "claude-3-5-sonnet-20240620"

llm = ChatAnthropic(model=anthropic_model)
# llm = ChatOpenAI(temperature=0, model="gpt-4o")
