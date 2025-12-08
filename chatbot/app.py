from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import streamlit as st 
import os
from dotenv import load_dotenv

## Load env from .env file
load_dotenv()

os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
#this feature is for langsmith tracking
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit app

st.title('LangChain Demo with Gemini API')
input_text=st.text_input('Enter the question for the bot')

## Gemini llm

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    response=chain.invoke(
        {
            'question':input_text
        }
    )
    st.write(response)
