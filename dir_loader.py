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
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ! PRIVATE KEYS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ! CONSTANTS
PDF_PATH = "documents/"
PERSIST_DB_PATH = "db2/"


# ! FUNCTIONS
def create_llm():
    return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")


def create_embeddings():
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)


def chroma_vector_store(docx: list[Document], embedx: embeddings, path: str):
    return Chroma.from_documents(docx, embedx, persist_directory=path)


if __name__ == "__main__":
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

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    prompt_template = """You are an assistant for question-answering tasks, use the following pieces retrieved to 
    answer the questions. If you are unable to obtain any answer from the context to any particular question, 
    you are free to make up your own answer suiting the situation.

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
