from __future__ import annotations
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

INDEX_NAME = "langchain-demo"
EMBED_MODEL = "models/embedding-001"
DIMENSION = 768
GEMMA_MODEL = "gemma-3-27b-it"

class ChatBot:
    def __init__(self) -> None:
        # Load and split PDF
        loader = PyPDFLoader(
            "./materials/ilide.info-viktor-frankl-man-s-search-for-meaning-pr_24dec9f5b7ce09386be953de1276f631.pdf"
        )
        pages = loader.load()
        docs = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        ).split_documents(pages)

        # Embed documents
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        # Build vector store
        docsearch = PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=INDEX_NAME,
        )

        # Configure LLM
        llm = ChatGoogleGenerativeAI(
            model=GEMMA_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_output_tokens=1024,
        )

        # Define prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a Toronto travel assistant. Use the context below to answer the user's question. "
                "If you don't know, say so. Keep it short and concise (≤ 2 sentences).\n\n"
                "Context: {context}\nQuestion: {question}\nAnswer:"
            ),
        )

        # Build RetrievalQA chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=docsearch.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
        )
