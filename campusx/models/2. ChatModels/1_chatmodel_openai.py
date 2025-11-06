from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini",temperature=0.3, max_completion_tokens=10)

result = model.invoke("What is the capital of India?") 

print(result.content)