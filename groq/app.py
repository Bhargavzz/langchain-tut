import os
import streamlit as st  
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

##load the groq api key

groq_api_key=os.environ["GROQ_API_KEY"]


st.title("Groq LLM with Langchain")


if "vector" not in st.session_state:
    st.session_state.embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
    st.session_state.loader=WebBaseLoader("https://docs.langchain.com/langsmith/create-account-api-key")
    st.session_state.docs=st.session_state.loader.load()
    
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    
    
    
    
llm=ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b"
    )
    
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
    
    """
    )
document_chain=create_stuff_documents_chain(llm,prompt)
    
retriever=st.session_state.vectors.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)
    
prompt=st.text_input("Enter your Prompt here:")
    
    
if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input": prompt})
       
    print("Response Time:",time.process_time()-start)
    st.write(response['answer'])
    
    
    with st.expander("Document Simialrity Search"):
        for idx,chunk in enumerate(response['context']):
            st.write(chunk.page_content)
        st.write("----------------------------------------")
            
            
        