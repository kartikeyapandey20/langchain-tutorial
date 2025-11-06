from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda , RunnableParallel
from langchain_groq import ChatGroq
import os

# Load environment variables from.env file
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Define the LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a movie critic"),
        ("human" , "Provide a brief summary of the {movie_name}.")
    ]
)


# Define plot analysis step
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system" , "You are a movie critic"),
            ("human" , "Analyze the plot of the movie {plot}.What are the strength and weaknesses?")
        ]
    )
    
    return plot_template.format_prompt(plot=plot)

# Define character analysis step
def analyze_character(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system" , "You are a movie critic"),
            ("human" , "Analyze the characters: of the movie {characters}.What are their strength and weaknesses?")
        ]
    )
    
    return character_template.format_prompt(characters=characters)

def combin_verdict(plot_analysis , character_analysis):
    return f"Plot Analysis: {plot_analysis}\n\nCharacter Analysis: {character_analysis}"
 
plot_branch_chain = (
    RunnableLambda(lambda x : analyze_plot(x)) | llm | StrOutputParser()
    )

character_branch_chain = (
    RunnableLambda(lambda x : analyze_character(x)) | llm | StrOutputParser()
    )

chain = (
    summary_template 
    |llm
    |StrOutputParser()
    |RunnableParallel(branches={"plot": plot_branch_chain,"character" : character_branch_chain})
    |RunnableLambda(lambda x: combin_verdict(x["branches"]["plot"],x["branches"]["character"]))
    
)

# Run the chain
result = chain.invoke({"movie_name" : "Inception"})

print(result)
