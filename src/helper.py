from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec



#loads the pdf file using langhchahain pdf and directory loaders 
#returns a list of document objects 
#one document object per page stored in a list. each object has metadata (source and pg#) and page content
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

#split the text into chunks
#returns a list of documents where each document is a chunk based on the page_content
def split_text(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    #split at the document level
    text_chunks= text_splitter.split_documents(extracted_data)
    return text_chunks

def download_huggingface_embeddings():
    #hugging face embedding model returns 384 dimensional vector
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

