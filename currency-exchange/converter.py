import os
from dotenv import load_dotenv
from fastapi import FastAPI
import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import json
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

load_dotenv()

query="convert 10 inr to usd dollar,use convert tool"


hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
currency_key = os.getenv("CURR_API")

url="https://api.currencyapi.com/v3/latest"


def conversion_rate(base_currency:str,currencies:str):
 response=requests.get(
    url,
    headers={"apikey":currency_key},
    params={"base_currency":base_currency.upper(),"currencies":currencies.upper()},
    timeout=15,
 )
 data=response.json()
 return data["data"][currencies.upper()]["value"]

@tool
def convert(base:str,curr:str,val:int):
   """Convert an amount from one currency to another.

    Args:
        base: The source currency code, such as INR.
        curr: The target currency code, such as USD.
        val: The amount to convert.
    """
   rate=conversion_rate(base,curr)
   return val*rate

tools=[convert]

llm_with_tools=llm.bind_tools(tools)

response=llm_with_tools.invoke(query)

if response.tool_calls:
   for tool_call in response.tool_calls:
     result=convert.invoke(tool_call["args"])
     print(result)
else:
     print("no tool used")