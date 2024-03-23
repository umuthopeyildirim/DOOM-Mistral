# GameCopilot
# Todo: Connect GROG with RAG
# Connect MongoDB with RAG

import embedding
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from decouple import config

GROG_API_KEY = config('GROG_API_KEY')

# MistralAI Embedding
embedding.embedQuery("Tell me a joke.")


# GROG API KEY
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768",
                api_key=GROG_API_KEY)


system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", human)])

chain = prompt | chat
for chunk in chain.stream({"text": "Tell me a joke."}):
    print(chunk.content, end="", flush=True)
