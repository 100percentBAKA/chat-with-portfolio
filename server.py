import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
from langchain.schema import embeddings
from langchain_astradb import AstraDBVectorStore
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define constants
PDF_PATH = "documents/"
PERSIST_DB_PATH = "db2/"
ASTRA_COLLECTION_NAME = "pdf_db"
ASTRA_NAMESPACE = "default_keyspace"
RESUME_PDF_PATH = "documents/resume_merged.pdf"

# ENV KEYS
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Create LLM
def create_llm():
    return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")


def create_ollama_llm():
    return ChatOllama(model="llama3")


def create_google_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-pro"
    )


# Create embeddings
def create_embeddings():
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)


def create_ollama_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


# Create Chroma vector store
def chroma_vector_store(docx: list[Document], embedx: embeddings, path: str):
    return Chroma.from_documents(docx, embedx, persist_directory=path)


def astra_vector_store(embedx: embeddings, collection_name: str, api_endpoint: str, token: str, namespace: str):
    return AstraDBVectorStore(
        embedding=embedx,
        collection_name=collection_name,
        api_endpoint=api_endpoint,
        token=token,
        namespace=namespace
    )


# Load documents and create vector store (executed once at startup)
llm = create_llm()
embeddings = create_embeddings()

# loader = DirectoryLoader(
#     PDF_PATH,
#     "**/*.pdf",
#     use_multithreading=True
# )

# ! LOADING THE DOCUMENTS
loader = PyPDFLoader(RESUME_PDF_PATH)

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
load_docs = loader.load_and_split(text_splitter=splitter)
vector_store = chroma_vector_store(load_docs, embeddings, PERSIST_DB_PATH)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

prompt_template = """You are Adarsh, and your full name is Adarsh G S. More details about you will be provided in the 
following context. You will be facing interview in a short while and I want you to be able to answer to your best. If 
any detail is missing in the context, its your responsibility to come with suitable solutions. If you are asked to 
provide with any weblinks, always ensure to provide the answers in markdown code format. Always try to be as much 
descriptive as possible, and always try your level best to convince people why I am the best for their company. Note 
that my github account link is https://www.github.com/100percentBAKA, which is different than the one in the context.

CONTEXT:
{context}

QUESTION: {question}

YOUR ANSWER:"""
prompt_template = ChatPromptTemplate.from_template(prompt_template)

# Create chain
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
)


# Define request model
class QueryRequest(BaseModel):
    query: str


# Define POST endpoint
@app.post("/query")
async def handle_query(request: QueryRequest, background_tasks: BackgroundTasks):
    query = request.query
    try:
        response = []
        for chunk in chain.stream(query):
            response.append(chunk)

        # Run vector_store.delete_collection() in the background
        # background_tasks.add_task(vector_store.delete_collection)

        return {"answer": "".join(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
