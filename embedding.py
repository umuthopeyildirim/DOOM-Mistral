# MistralAI Embedding
from langchain_mistralai import MistralAIEmbeddings
from decouple import config

MISTRAL_API_KEY = config('MISTRAL_API_KEY')
embedding = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
embedding.model = "mistral-embed"  # or your preferred model if available


def embedQuery(query):
    return embedding.embed(query)


def embedDocuemt(key, value):
    # Example: res_document = embedding.embed_documents(["test1", "another test"])
    return embedding.embed_documents([key, value])
