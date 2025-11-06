from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_apk_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_apk_key, model_name="Llama3-8b-8192")

result = llm.invoke("What is square root of 49")

print(result.content)