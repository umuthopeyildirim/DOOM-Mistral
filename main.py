from huggingface_hub import InferenceClient
import gradio as gr
from langchain_groq import ChatGroq
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
client = ChatGroq(api_key=os.getenv("FIREWORKS_API_KEY")),


mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = "gamecopilot"
vector_search_collection = "all"

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    os.getenv("MONGO_URI"),
    db + "." + vector_search_collection,
    MistralAIEmbeddings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-embed"),
)


def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt


def generate(prompt, history, temperature=0.9, max_new_tokens=256):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2

    generate_kwargs = dict(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    results = vector_search.similarity_search(prompt)

    # Display results
    for result in results:
        print("Results: ", result)

    formatted_prompt = format_prompt(prompt, history)

    stream = client.stream(
        formatted_prompt, **generate_kwargs)
    output = ""

    for chunk in stream:
        output += chunk.content
        yield output
    return output


additional_inputs = [
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    )
]


gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False,
                       show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
    title="""GameCopilot""",
    description="""GameCopilot is a chatbot that helps you to find the best games for you. Powered by Groq, MistralAI(Embeddings) and MongoDB Atlas.""",
).launch(show_api=False)
