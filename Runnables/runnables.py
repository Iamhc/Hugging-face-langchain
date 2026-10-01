from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="novita",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(template="get me random topic in science not known well known",input_variables=[])

prompt2=PromptTemplate(template="tell random fact about {text}",input_variables=["text"])

parser=StrOutputParser()

chain=prompt1 | model | parser | RunnableParallel({
    "topic":RunnablePassthrough(),
    "fact": prompt2 | model | parser
}
)


output=chain.invoke({})

print(output)