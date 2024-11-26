import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from helper_functions import encode_pdf
from evalute_rag import *

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '')))

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define the cross encoder class
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

from pydantic import BaseModel, Field


class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")


def get_vector_store(vectorstore):
    cross_encoder_retriever = CrossEncoderRetriever(
        vectorstore=vectorstore,
        cross_encoder=cross_encoder,
        k=20,
        rerank_top_k=15
    )
    return cross_encoder_retriever


def get_response(cross_encoder_retriever, user_question):
    llm_cross_encoder = ChatOpenAI(temperature=0, model_name="gpt-4o")
    qa_chain_cross_encoder = RetrievalQA.from_chain_type(
        llm=llm_cross_encoder,
        chain_type="stuff",
        retriever=cross_encoder_retriever,
        return_source_documents=True
    )
    result_cross_encoder = qa_chain_cross_encoder({"query": user_question})
    return result_cross_encoder


# Main application logic
def main():
    st.header("📄 Chat with your PDF file 🤗")

    # Initialize session state for vectorstore and retriever
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "cross_encoder_retriever" not in st.session_state:
        st.session_state.cross_encoder_retriever = None

    # Step 1: Upload PDF and preprocess
    if st.session_state.vectorstore is None:
        pdf_docs = st.file_uploader("Upload your PDF file and click on the Process button", accept_multiple_files=False)

        if pdf_docs:
            st.write("Processing file:", pdf_docs.name)
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(pdf_docs.getvalue())
            st.session_state.vectorstore = encode_pdf(temp_file)
            st.session_state.cross_encoder_retriever = get_vector_store(st.session_state.vectorstore)
            st.success("File processed and ready for queries!")

    # Step 2: Handle user query
    if st.session_state.cross_encoder_retriever is not None:
        query = st.text_input("Ask questions about your uploaded PDF file")
        if query:
            response = get_response(st.session_state.cross_encoder_retriever, query)['result']
            st.write(response)
    elif st.file_uploader:
        st.warning("Please upload and process a PDF file first.")

if __name__ == "__main__":
    main()
