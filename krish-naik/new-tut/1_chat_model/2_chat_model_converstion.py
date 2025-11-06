from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_apk_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_apk_key, model_name="Llama3-8b-8192")


message  = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engaging posts on instagram")
]

result  = llm.invoke(message)

print(result.content)