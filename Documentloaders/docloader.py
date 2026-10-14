from langchain_community.document_loaders import PyPDFLoader

pdf=PyPDFLoader("BI_Lab_Manual.pdf")
print(pdf.load()[0])