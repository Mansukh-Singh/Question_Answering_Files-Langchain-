from fastapi import FastAPI, File, UploadFile, Depends
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain 
from langchain.vectorstores.pinecone import Pinecone
from backend.functions import split_data, data_injestion, retrieval_QA
from backend.pinecone_helper import get_documents, existing_index
from dotenv import load_dotenv
import pinecone
import os 

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

llm = OpenAI(temperature=0.6)

docs_folder_path = "backend_docs"

if not os.path.exists(docs_folder_path):
    os.makedirs(docs_folder_path)

@app.post("/upload_file/")
def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(docs_folder_path, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return {"result":"File uploaded"}

@app.post("/insert_data_into_vector_database/")
def insert_data():
    if os.path.exists(docs_folder_path) and os.path.isdir(docs_folder_path):
        files_in_docs = os.listdir(docs_folder_path)
        if files_in_docs:
            file = files_in_docs[0]
            docs = split_data(f"{docs_folder_path}/{file}")
            data_injestion(docs)
            return {"file":"data_injested"}        
        else:
            return {"error":"The 'docs' folder is empty."}
    else:
        return {"error":"No docs folder exists"}

@app.post("/query/")
def query(text: str):
    vector_docs = existing_index()
    chain = retrieval_QA(vector_docs)
    answer = chain({"question":text}, return_only_outputs = True)
    return {"answer": answer}    





