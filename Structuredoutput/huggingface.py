from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="novita",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

output=input("hi prompt please ")


parser=JsonOutputParser()

template=PromptTemplate(
    template="answer the query {query} in {instructions}",
   input_variables=["query"],
   partial_variables={"instructions":parser.get_format_instructions()}
)

chain=template | model | parser
r=chain.invoke({"query":output})
print(r)