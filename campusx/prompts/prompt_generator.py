from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
    \nPlease summarize the research paper titled \"{paper_input}\" with the following specifications:\nExplanation Style: {style_input}  
    \nExplanation Length: {length_input}  \n1. Mathematical Details:  \n   - Include relevant mathematical equations if present in the paper. 
    \n   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  \n2. Analogies:  
    \n   - Use relatable analogies to simplify complex ideas.  
    \nIf certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing. 
    \nEnsure the summary is clear, accurate, and aligned with the provided style and length.\n
    """,
    input_variables=["paper_input","style_input","length_input"]
)

template.save("template.json")