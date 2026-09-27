from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="novita",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

output=input("hi prompt please ")
response=model.invoke(output)

parser=StrOutputParser()

res=parser.invoke(response)
print(res)