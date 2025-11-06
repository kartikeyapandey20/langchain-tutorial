from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


chat_template = ChatPromptTemplate([
    ("system" , "you are a helpful {domain} Expert"),
    ("human", "Explain in simple terms what is {topic}")
])

prompt = chat_template.invoke({
    "domain" : "AI/ML",
    "topic" : "Machine Learning"
})

print(prompt)  