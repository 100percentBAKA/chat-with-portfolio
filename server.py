import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import embeddings
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define constants
PDF_PATH = "documents/"
PERSIST_DB_PATH = "db2/"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Create LLM
def create_llm():
    return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")


# Create embeddings
def create_embeddings():
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)


# Create Chroma vector store
def chroma_vector_store(docx: list[Document], embedx: embeddings, path: str):
    return Chroma.from_documents(docx, embedx, persist_directory=path)


# Load documents and create vector store (executed once at startup)
llm = create_llm()
embeddings = create_embeddings()

loader = DirectoryLoader(
    PDF_PATH,
    "**/*.pdf",
    use_multithreading=True
)

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
load_docs = loader.load_and_split(text_splitter=splitter)
vector_store = chroma_vector_store(
    docx=load_docs,
    embedx=embeddings,
    path=PERSIST_DB_PATH
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

prompt_template = """You are an assistant for question-answering tasks, use the following pieces retrieved to 
    answer the questions. If you are unable to obtain any answer from the context to any particular question, 
    you are free to make up your own answer suiting the situation. Always ensure to provide the answers in markdown 
    code format, is the answer is to consist any web links or code section

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
async def handle_query(request: QueryRequest):
    query = request.query
    try:
        response = []
        for chunk in chain.stream(query):
            response.append(chunk)
        return {"answer": "".join(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
