from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_apk_key = os.getenv('GROQ_API_KEY')

model = ChatGroq(groq_api_key=groq_apk_key, model_name="Llama3-8b-8192")

chat_history = []

system_mesaage = SystemMessage("You are any helpful AI assistant.")
chat_history.append(system_mesaage)

#Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    
    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    print(f"AI : {response}")
    
print("----Message History----")
print(chat_history)