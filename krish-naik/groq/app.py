import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

# load the GROQ API KEY
groq_apk_key = os.getenv('GROQ_API_KEY')

st.title('Chatgroq with Llama3 Demo')

llm = ChatGroq(groq_api_key=groq_apk_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    
    """
)

def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        # Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        # Document  Loading
        st.session_state.docs = st.session_state.loader.load()
        # Chunk creation
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Splitter
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        # Vector Ollama Embedding
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1 = st.text_input("Enter your question from Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vectore Store DB is Ready")
    

if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm , prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriveval_chain = create_retrieval_chain(retriever,document_chain)
    response = retriveval_chain.invoke({'input':prompt1})    
    st.write(response['answer'])
    
    # with a streamlist expander
    with st.expander("Document Similarity"):
        # Find the relevant chunks
        for i , doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------")
            