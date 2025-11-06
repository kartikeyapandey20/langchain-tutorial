from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are
oo many pre-installed apps that I can't remove. Also, the UI looks outdated compared to
ther brands. Hoping for a software update to fix this.""")

print(type(result))
print(result['summary'])
print(result['sentiment'])