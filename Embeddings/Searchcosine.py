from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
embeddings=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

docs=["spanish is a language",
"hindi is a language",
"english is a language",
"Maths is a subject",
"geometry is a part of higher maths"]

doc_embeds=embeddings.embed_documents(docs)

search="geometry and maths"

search_embeds=embeddings.embed_query(search)

cosineresults=cosine_similarity([search_embeds],doc_embeds)[0]

index,similarity=sorted(list(enumerate(cosineresults)),key=lambda x:x[1])[-1]

print(docs[index])