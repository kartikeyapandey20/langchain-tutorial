from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
import os

# Load environment variables from.env file
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Define the LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a factory facts expert who knows fact about {animal}"),
        ("human" , "Tell me {fact_count} facts.")
    ]
)

# Create the combined chain using Langchain Expressions Language

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"animal":"elephant" , "fact_count" : 1})

print(result)