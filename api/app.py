from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import 
from langserve import add_routes
from langchain_community.llms.ollama import Ollama
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

app = FastAPI(
    title="Langchain server",
    version="1.0",
    description="A simple API sever"
)

# ollama
llm = Ollama(model='llama3.2:1b')

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")


add_routes(
    app,
    prompt1|llm,
    path='/essay'
)

add_routes(
    app,
    prompt2|llm,
    path='/poem'
)

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)