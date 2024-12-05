import requests
import streamlit as st

def get_llm_essay_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
    json={"input" : {'topic' : input_text}})
    print(response.json())
    return response.json()['output']


def get_llm_poem_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
    json={"input" : {'topic' : input_text}})
    
    return response.json()['output']

st.title('Langchain demo with ollama API')
input_text_essay = st.text_input("Write essay on")
input_text_poem = st.text_input("Write poem on")

if input_text_essay:
    st.write(get_llm_essay_response(input_text_essay))
    
if input_text_poem:
    st.write(get_llm_poem_response(input_text_poem))