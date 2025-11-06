from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
# langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


# Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistent.Please response to the user query"),
        ("user","Question:{question}")
    ]
)

# streamlit framework
st.title("Chatbot with llama3.2:1b")
input_text = st.text_input("Search The Topic you want")

# ollama llama3.2:1b
llm = ollama.Ollama(model='llama3.2:1b')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))