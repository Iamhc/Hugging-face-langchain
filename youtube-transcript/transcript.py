from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


ytt_api = YouTubeTranscriptApi()
res=ytt_api.fetch('abiL84EAWSY')

ans=""

for input in res:
    ans+=input.text


text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)

chunks=text_splitter.create_documents([ans])

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


query="was sky discussed in anywhere, if yes then what was the context"

db=FAISS.from_documents(chunks,embeddings)

final_prompt=db.similarity_search(query,k=3)

context="\n\n".join(doc.page_content for doc in final_prompt)
final_ans=llm.invoke(f"""
You are answering questions about a YouTube transcript.

Use only the retrieved context below.
Do not invent quotations, events, or interpretations.

Find the exact passage relevant to the question.
Return the complete paragraph containing the relevant line,
including the sentences before and after it, as available
in the retrieved context.

Preserve the original wording. Do not paraphrase or create
sentences that are not present in the context.

If the context does not contain a relevant passage, say:
"I couldn't find sufficient evidence in the retrieved transcript."

Question:
{query}

Retrieved context:
{context}

Answer:
""")

print(final_ans.content)