from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs=[
    "time",
    "displacement"
]

res=embeddings.embed_documents(docs)

print(res)