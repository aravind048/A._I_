from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def build_rag_chain(documents):
    # Use local embeddings from sentence-transformers
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Load LLM from HuggingFace
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",  # Or other instruct model
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    return qa_chain
