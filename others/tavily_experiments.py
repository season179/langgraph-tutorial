import os
from tavily import TavilyClient
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

question = "What is the AlphaCodium paper about?"
# For basic search:
# response = tavily.search(query=question)
# print(f"1: {response}\n---\n")

# # For advanced search:
# response = tavily.search(query=question, search_depth="advanced")
# # Get the search results as context to pass an LLM:
# context = [{"url": obj["url"], "content": obj["content"]} for obj in response["results"]]
# print(f"2: {context}\n---\n")

# # You can easily get search result context based on any max tokens straight into your RAG.
# # The response is a string of the context within the max_token limit.
# response = tavily.get_search_context(query=question, search_depth="advanced", max_tokens=1500)
# print(f"3: {response}\n---\n")

# You can also get a simple answer to a question including relevant sources all with a simple function call:
# tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
response = tavily.qna_search(query=question, search_depth="advanced")
print(f"4: {response}\n---\n")
