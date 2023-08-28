from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
import openai
import gradio as gr
import os
import sys

os.environ['OPENAI_API_KEY'] = '--your key here --' # noqa


def construct_index(directory_path):
    num_outputs = 512

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    index.storage_context.persist('index.json')

    return index

index = construct_index("docs")

def chatbot(input_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
#    index = GPTVectorStoreIndex.load_from_disk('index.json')
#    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.Textbox(lines=7, label="Enter your text:"),
                     outputs="text",
                     title="SoloGPT AI Chatbot")

iface.launch(share=True)
