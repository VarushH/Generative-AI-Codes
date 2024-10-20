from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()


#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Questions: {question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):

    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer


#Title of the app
st.title("Custom Search Engine")

#Sidebar for settings
st.sidebar.title("Settings")

#Dropdown to select various OpenAI models
llm = st.sidebar.selectbox("Select a Ollama Model",["mistral","llama3.2","gemma2"])

#Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

#User Input

st.write("Kindly drop your questions here")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)

# else:
#     st.write("Please provide the query")

