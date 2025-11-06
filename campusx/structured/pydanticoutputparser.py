from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Optional

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

    
class Person(BaseModel):
    
    name: str = Field(description="A Name of the fictional Charactor")
    city: str = Field(description="A City of the fictional Charactor")
    age: int = Field(description="A age of the fictional Charactor")
    
    
model = ChatHuggingFace(llm=llm)
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me name , age and city of a fiction {place}person \n{format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'Indian'})
print(result)