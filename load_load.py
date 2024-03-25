from pymongo import MongoClient
from langchain_mistralai import MistralAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Set the MongoDB URI, DB, Collection Names
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = MongoClient(os.getenv("MONGO_URI"))
dbName = "gamecopilot"
collectionName = "all"
collection = client[dbName][collectionName]

# Initialize the DirectoryLoader
loader = DirectoryLoader('./sample_files', glob="./*.txt", show_progress=True)
data = loader.load()

# Define the OpenAI Embedding Model we want to use for the source data
# The embedding model is different from the language generation model
embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)

# Initialize the VectorStore, and
# vectorise the text from the documents using the specified embedding model, and insert them into the specified MongoDB collection
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    data, embeddings, collection=collection)
