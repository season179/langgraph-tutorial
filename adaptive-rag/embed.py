from langchain_nomic.embeddings import NomicEmbeddings

embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")