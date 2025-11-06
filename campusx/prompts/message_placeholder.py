from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are helpful customer support agent'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human','Where is my refund?')
])

chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)


chat_template.invoke({
    'chat_history' : chat_history,
    'query' : 'where is my refund'
})

print(chat_template)