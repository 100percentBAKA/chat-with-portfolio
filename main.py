import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import embeddings
from langchain.schema import Document
from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ! PRIVATE KEYS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")

# ! CONSTANTS
RESUME_PDF_PATH = "documents/resume.pdf"
CHROMA_DB_PATH = "./db"
ASTRA_COLLECTION_NAME = "pdf_db"


# ! FUNCTIONS
def create_llm(selection: str):
    if selection == "Y":
        return ChatOllama(model="llama3")
    else:
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")


def create_embeddings(selection: str):
    if selection == "Y":
        return OllamaEmbeddings(model="nomic-embed-text")
    else:
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)


def chroma_vector_store(docx: list[Document], embedx: embeddings, path: str):
    return Chroma.from_documents(docx, embedx, persist_directory=path)


def astra_vector_store(embedx: embeddings, collection_name: str, api_endpoint: str, token: str):
    return AstraDBVectorStore(
        embedding=embedx,
        collection_name=collection_name,
        api_endpoint=ASTRA_DB_ENDPOINT,
        token=ASTRA_TOKEN,
    )


if __name__ == "__main__":
    LOCAL_PERSISTENCE = input("Use Astra DB (y/[N]):") or "N"
    LLM_SELECTION = input("Use Llama3:7B from Ollama ([Y]/n): ") or "Y"
    EMBEDDING_SELECTION = input("Use Nomic-embed-text from Ollama ([Y]/n): ") or "Y"

    # ! CREATING AN LLM
    llm = create_llm(LLM_SELECTION)

    # ! CHOOSING EMBEDDINGS MODEL
    embeddings = create_embeddings(EMBEDDING_SELECTION)

    # ! LOADING THE DOCUMENTS
    loader = PyPDFLoader(RESUME_PDF_PATH)

    # ! LOAD THE SPLITTER
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    # ? chunk_size=512

    # ! SPLIT THE DOCUMENT INTO CHUNKS
    docs_from_pdf = loader.load_and_split(text_splitter=splitter)

    # ! CHOOSING VECTOR STORE
    if LOCAL_PERSISTENCE == "N":
        vector_store = chroma_vector_store(docs_from_pdf, embeddings, CHROMA_DB_PATH)
    else:
        vector_store = astra_vector_store(embeddings, ASTRA_COLLECTION_NAME, ASTRA_DB_ENDPOINT, ASTRA_TOKEN)

    # ! CREATING A RETRIEVER
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2})  # ? search_kwargs specifies the number of similar chunks/docs to return

    # ! CREATING A TEMPLATE
    prompt_template = """You are an assistant for question-answering tasks, use the following pieces retrieved to 
    answer the questions. If you are unable to obtain any answer from the context to any particular question, 
    you are free to make up your own answer suiting the situation. Always ensure to provide the answers in markdown 
    code format, is the answer is to consist any web links or code section

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:"""

    prompt_template = ChatPromptTemplate.from_template(prompt_template)

    # ! CREATING A CHAIN
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )

    # ! TAKING THE QUERY FROM THE USER
    query = input("Enter the query: ")

    # ! IMPLEMENTING A STREAM
    for chunk in chain.stream(query):
        print(chunk, end="", flush=True)
