import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def get_llm():
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return ChatGroq(model="openai/gpt-oss-20b", api_key=groq_key)
    return ChatOllama(model="llama3.1")