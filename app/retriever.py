from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_vectordb(persist_directory="db"):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
