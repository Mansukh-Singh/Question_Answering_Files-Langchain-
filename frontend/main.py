import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import pickle, time

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.6)

st.title("Q and A")
st.sidebar.title("File Upload Section")

main_placeholder = st.empty()

uploaded_file = st.sidebar.file_uploader("Choose a pdf file")

docs_folder_path = "frontend_docs"

file_path_pickle = "Vector_Pickle.pkl"

if not os.path.exists(docs_folder_path):
    os.makedirs(docs_folder_path)

if uploaded_file:
    file_path = os.path.join(docs_folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(f"{docs_folder_path}/{uploaded_file.name}")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n','\n','.',','],
        chunk_size = 500
    )
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vector_docs = FAISS.from_documents(
        docs, 
        embeddings
    )
    time.sleep(2)


query = main_placeholder.text_input("Question: ")
if query:
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm = llm, 
        retriever = vector_docs.as_retriever()
    )
    answer = chain({"question":query}, return_only_outputs = True)['answer']
    st.markdown(answer)