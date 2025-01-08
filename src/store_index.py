import os
from dotenv import load_dotenv
from langchain_pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore  

from src.helper import download_huggingface_embeddings, load_pdf_file, split_text


#this file needs to be executed first time if add or remove data

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file("Data/")
#text chunks is a list of broken up documents
text_chunks = split_text(extracted_data)
embeddings = download_huggingface_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"
#only execute first time

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


#embed each chunk and upsert into pinecone index  
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks, 
    index_name=index_name, 
    embedding = embeddings
)

