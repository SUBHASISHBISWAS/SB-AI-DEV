from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model=ChatGroq(model="Llama3-8b-8192",groq_api_key=groq_api_key)

# Create ChatPromptTemplate
generic_template = "Translate the following into {language}"
prompt = ChatPromptTemplate.from_messages([("system",generic_template), ("user","{text}")])

parser = StrOutputParser()

# Create Chain

chain=prompt|model|parser

app=FastAPI(title="Langchain Server",Version="1.0", description="This is a FastAPI server for Langchain")

# Add routes
add_routes(app, chain,path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)