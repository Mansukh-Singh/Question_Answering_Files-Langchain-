from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from backend.pinecone_helper import create_index, data_injest
from dotenv import load_dotenv
import os 

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.6)

def split_data(file_name):
    loader = PyPDFLoader(file_name)
    data = loader.load()
    final_docs = split_data_into_chunks(data)
    return final_docs

def split_data_into_chunks(dataset):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 0
    )
    docs_data = text_splitter.split_documents(dataset)
    return docs_data

def data_injestion(docs_split):
    create_index()
    data_injest(docs_split)

def retrieval_QA(vector_array):
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm = llm, 
        retriever = vector_array.as_retriever()
    )
    return chain