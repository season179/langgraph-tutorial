from langchain_community.chat_models import ChatOllama


llm = ChatOllama(model="mistral:7b-instruct-q8_0", temperature=0.0)
# llm = ChatOllama(model="llama3:8b-instruct-q8_0", format="json", temperature=0.0)
