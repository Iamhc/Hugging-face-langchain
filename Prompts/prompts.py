from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

@st.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 200,
                         "return_full_text": False,},
        
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

st.header("Model")
template=PromptTemplate.from_template(
   "hi write cover letter for {role} in {tone} write it within 100 words. mention technolgies of {role}."
)

role=st.selectbox("choose one",["sde","product designer"])
tone=st.selectbox("choose one",["happy","respect and formal"])



if st.button("run"):
    chain = template | model
    with st.spinner("Generating..."):
        res = chain.invoke({"role": role, "tone": tone})
    st.write(repr(res.content))
    st.write(res)

