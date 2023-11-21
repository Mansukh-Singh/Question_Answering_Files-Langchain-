import os

import pinecone
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from fastapi import HTTPException

load_dotenv()

index_name = os.getenv('PINECONE_INDEX')

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(
    model = model_name
)

text_field = "text"
index = pinecone.Index(index_name)
vectorstore = Pinecone(index, embeddings.embed_query, text_field)

pinecone.init(
    api_key = os.getenv('PINECONE_API_KEY'),
    environment = os.getenv('PINECONE_ENV')
)

def get_documents(query):
    try:
        index = pinecone.Index(index_name)
        vectorstore = Pinecone(index, embeddings.embed_query, text_field)
        return vectorstore.similarity_search(
            query
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def create_index():
    for i in pinecone.list_indexes():
        pinecone.delete_index(i)
    pinecone.create_index(
        name=os.getenv('PINECONE_INDEX'), 
        dimension=int(os.getenv('PINECONE_DIMENSION')), 
        metric=os.getenv('PINECONE_METRIC')
    )

def data_injest(docs):
    Pinecone.from_documents(
        docs,
        embeddings,
        index_name = os.getenv('PINECONE_INDEX')
    )

def existing_index():
    vector_docs = None
    if index_name in pinecone.list_indexes():
        vector_docs = Pinecone.from_existing_index(
            index_name = os.getenv("PINECONE_INDEX"),
            embedding = embeddings,
        )
    return vector_docs