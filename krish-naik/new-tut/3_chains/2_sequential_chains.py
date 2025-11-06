from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_groq import ChatGroq
import os

# Load environment variables from.env file
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Define the LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
animal_fact_template = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a factory facts expert who knows fact about {animal}"),
        ("human" , "Tell me {fact_count} facts.")
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system" , "you are a translator and convertthe porvided text into {language}"),
        ("human","Translate the following text into {language}:{text}")
    ]
)

count_words = RunnableLambda(lambda x: f" Word counts : {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output:{"text" : output, "language":"french"})
# Create the combined chain using Langchain Expressions Language

chain = animal_fact_template | llm | StrOutputParser() | prepare_for_translation | translation_template | llm | StrOutputParser()

result = chain.invoke({"animal":"cat" , "fact_count" : 2})

print(result)