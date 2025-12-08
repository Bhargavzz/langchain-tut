import requests
import streamlit as st


def get_25flash_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
                           json={
                               "input":{
                                   "topic":input_text
                               }
                           })
    
    return response.json()['output']['content']



def get_20flash_response(input_text):
    response=requests.post("http://localhost:8000/poem/invoke",
                           json={
                               "input":{
                                   "topic":input_text
                               }
                           })
    
    return response.json()['output']['content']

st.title('LangChain Demo with Gemini API via FastAPI Server')

input_text=st.text_input('Enter the topic for essay (100 words)')
input_text1=st.text_input('Enter the topic for poem (10 lines)')


if input_text:
    response=get_25flash_response(input_text)
    st.write(response)

if input_text1:
    response=get_20flash_response(input_text1)
    st.write(response)