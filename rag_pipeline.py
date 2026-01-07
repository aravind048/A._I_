from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from hf_llm import hf_llm


def build_rag_chain(documents):
    """
    Builds a Retrieval-Augmented Generation (RAG) chain
    using FAISS + HuggingFace LLM (LangChain 1.2.0 compatible)

    Input:
        documents: List[Document]
    Output:
        Runnable chain (invoke with a string question)
    """

    # Use local embeddings from sentence-transformers
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Vector store
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 3. Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3})

    # 4. Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a factual question-answering assistant.

            Answer the question using ONLY the provided context.
            Follow these rules strictly:
            - If the question asks for a specific fact, give ONLY that fact.
            - Do NOT add extra explanation.
            - Do NOT infer beyond the context.
            - If the answer is not explicitly present, say: "Not found in document."

            Context:
            {context}

            Question:
            {question}
        """
    )

    # 7. Chain (THIS replaces RetrievalQA)
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | hf_llm
    )

    return rag_chain
