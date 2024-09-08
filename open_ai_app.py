from langchain_core import output_parsers
import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with OPENAI"

## prompt template
prompt=ChatPromptTemplate.from_messages(
  [
  ("system", "You are a helpful assistance. Please response to the user queries"),
  ("user", "Question:{question}"),
  ]
)

def generate_response(question, api_key, engine, temprature, max_tokens):
  openai.api_key=api_key
  llm=ChatOpenAI(model=engine)
  output_parser=StrOutputParser()
  chain=prompt|llm|output_parser
  answer=chain.invoke({"question":question})
  return answer

#title
st.title("Enhances QandA Chatbot with OpenAI")
#Sidebar for Settings
st.sidebar.title("Settings")
#to set the spi key
api_key=st.sidebar.text_input("Enter your OpenAI API key:", type="password")
#to select the llm model
llm=st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
#to select the temperature
temprature=st.sidebar.slider("Temprature", min_value=0.0, max_value=1.0, value=0.7)
#to select the maximum number of tokens
max_token=st.sidebar.slider("Maximum Tokens", min_value=50, max_value=300, value=150)

#main interface
st.write("Ask any question!")
user_input=st.text_input("Enter your question")

if user_input:
  response=generate_response(user_input, api_key, llm, temprature, max_token)
  st.write(response)
else:
  st.write("Please ask any question first!")






