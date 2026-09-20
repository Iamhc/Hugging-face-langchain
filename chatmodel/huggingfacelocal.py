from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 200}
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("tell about mlops")
print(response.content)