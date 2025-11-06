from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

groq_apk_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_apk_key, model_name="Llama3-8b-8192")

template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone": "formal",
    "company": "Google",
    "position": "Software Engineer",
    "skill": "Python"
})

result = llm.invoke(prompt)
print(result.content)