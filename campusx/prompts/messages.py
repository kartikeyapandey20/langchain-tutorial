from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

messsages = [
    SystemMessage(content="You are a helpful asistant"),
    HumanMessage(content="tell me about transformer")
]

result = model.invoke(messsages)

messsages.append(AIMessage(content=result.content))

print(messsages)