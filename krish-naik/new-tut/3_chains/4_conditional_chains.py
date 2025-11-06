from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
import os

load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')
model = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)

# Define promopt template for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","Generate a thank you note for this positive feedback: {feedback}"),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","Generate a thank you note for this negative feedback: {feedback}"),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","Generate a thank you note for this neutral feedback: {feedback}"),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","Generate a message to escalate this feedback to human agent: {feedback}"),
    ]
)

classification_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","Classify the sentiment of this feedback as positive , negative , neutral, or escalate: {feedback}"),
    ]
)

branches = RunnableBranch(
    (
        lambda x : "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x : "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x : "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model  | StrOutputParser()
)

classification_feedback_template = classification_feedback_template | model | StrOutputParser() 

chain = classification_feedback_template | branches

review = "The product is terrible. It broke after just one use and the quality is very poor"

result  = chain.invoke({"feedback": review})

print(result)