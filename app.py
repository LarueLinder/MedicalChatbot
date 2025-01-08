import os
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_huggingface_embeddings
from src.prompt import *



app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

index_name = "medicalchatbot"
embeddings = download_huggingface_embeddings()

#load the existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, 
    embedding = embeddings
)

#k=3 so retriever will return 3 revelent document objects based on similarity
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=500
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
#rag chain uses the retriever object previously defined to retrieve relevant documents 
#then uses the question_answer chain which is a chain w/ the llm and prompt
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": input})
    print("Response: ", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":  
    app.run(host="0.0.0.0", port=8080, debug = True)    