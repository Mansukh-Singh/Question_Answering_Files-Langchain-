from fastapi import FastAPI, File, UploadFile, Depends
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain 
from langchain.vectorstores.pinecone import Pinecone
from backend.functions import split_data, data_injestion
from backend.pinecone_helper import get_documents, existing_index
from dotenv import load_dotenv
import pinecone
import os 

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

shared_data = {"value": None}

global_vector_docs = None

llm = OpenAI(temperature=0.6)

def get_shared_data():
    return shared_data

docs_folder_path = "docs"

if not os.path.exists(docs_folder_path):
    os.makedirs(docs_folder_path)

@app.post("/upload_file/")
def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(docs_folder_path, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return {"result":"File uploaded"}

@app.post("/insert_data_into_vector_database/")
def insert_data(shared_data: dict = Depends(get_shared_data)):
    if os.path.exists(docs_folder_path) and os.path.isdir(docs_folder_path):
        files_in_docs = os.listdir(docs_folder_path)
        if files_in_docs:
            global global_vector_docs
            file = files_in_docs[0]
            docs = split_data(f"{docs_folder_path}/{file}")
            vector_docs = data_injestion(docs)
            global_vector_docs = vector_docs
            return {"file":"data_injested"}        
        else:
            return {"error":"The 'docs' folder is empty."}
    else:
        return {"error":"No docs folder exists"}

@app.post("/query/")
def query(text: str, shared_data: dict = Depends(get_shared_data)):
    docs = get_documents(text)
    vector_docs = existing_index()
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm = llm, 
        retriever = vector_docs.as_retriever()
    )
    answer = chain({"question":text}, return_only_outputs = True)
    return {"answer": answer}    





