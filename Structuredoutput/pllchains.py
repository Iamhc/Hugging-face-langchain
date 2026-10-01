from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from operator import itemgetter


load_dotenv()

parser=StrOutputParser()

llm=HuggingFaceEndpoint(
   repo_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="novita",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm) 

topic=PromptTemplate.from_template("prepare a topic random on tech") | model | parser

quiz=PromptTemplate.from_template("Get questions on {topic}") | model | parser

pllchain=RunnableParallel({
"info":itemgetter("topic"), # info assigned here"""
"quiz": quiz
}
)

prompt3=PromptTemplate.from_template(
    "Get sols with those questions on {info} with questions {quiz}"
)

chain={"topic":topic } | pllchain | prompt3 | model | parser 
ans=chain.invoke({})
# "topic"  assigned here  """

chain.get_graph().print_ascii()
print(ans)
