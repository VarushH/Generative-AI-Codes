from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

generic_template = "Transalate the following from {from_language} to {to_language}"
prompt = ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")])

parser = StrOutputParser()

#Create Chain
chain = prompt|model|parser

#App definition

app = FastAPI(title="Langchain Server",version="1.0",description="A simple API server using Langchain runnable interface")
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    
    uvicorn.run(app,host="127.0.0.1",port=8000)


"""
Guide to view the working:

Swagger Link - http://127.0.0.1:8000/docs#/

To call one of the models - http://127.0.0.1:8000/docs#/chain/chain_invoke_chain_invoke_post


Fill in the inputs in /chain/invoke as below

{
  "input": {
    "from_language": "English",
    "text": "hello",
    "to_language": "French"
  },
  "config": {},
  "kwargs": {}
}


http://127.0.0.1:8000/chain/input_schema

Test any API's from Swagger Link above in Postman for testing
"""