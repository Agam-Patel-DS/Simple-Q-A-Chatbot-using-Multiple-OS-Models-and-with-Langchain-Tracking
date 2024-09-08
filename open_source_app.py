from langchain_core import output_parsers
import streamlit as st
import openai
#from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with Open Source Models"
groq_api_key=os.getenv("GROQ_API_KEY")

## prompt template
prompt=ChatPromptTemplate.from_messages(
  [
  ("system", "You are a helpful assistance. Please response to the user queries"),
  ("user", "Question:{question}"),
  ]
)

def generate_response(question, engine, temprature, max_tokens):
  llm= ChatGroq(groq_api_key=groq_api_key, model_name=engine) #Llama3-8b-8192
  output_parser=StrOutputParser()
  chain=prompt|llm|output_parser
  answer=chain.invoke({"question":question})
  return answer

#title
st.title("Enhances QandA Chatbot with Open Source Models")
#Sidebar for Settings
st.sidebar.title("Settings")
#to select the llm model
llm=st.sidebar.selectbox("Select an Open Source Model Model", ["gemma2-9b-it", "Llama3-8b-8192", "mixtral-8x7b-32768"])
#to select the temperature
temprature=st.sidebar.slider("Temprature", min_value=0.0, max_value=1.0, value=0.7)
#to select the maximum number of tokens
max_token=st.sidebar.slider("Maximum Tokens", min_value=50, max_value=300, value=150)

#main interface
st.write("Ask any question!")
user_input=st.text_input("Enter your question")

if user_input:
  response=generate_response(user_input, llm, temprature, max_token)
  st.write(response)
else:
  st.write("Please ask any question first!")






