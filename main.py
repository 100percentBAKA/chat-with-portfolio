from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.schema import embeddings
from langchain.schema import vectorstore

from dotenv import load_dotenv

load_dotenv()

RESUME_PDF_PATH = "documents/resume.pdf"
CHROMA_DB_PATH = "./db"

# llm = ChatOllama(
#     model="llama3:latest"
# )

llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)


def load_document(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


def split_documents(docx: list[Document]):
    print("Splitting documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    '''
        add_start_index=True will ensure that the character at the end of each split, is also included in the next split
    '''
    all_splits = text_splitter.split_documents(docx)
    print("Splitting documents done")
    return all_splits


def to_vector_store(docx: list[Document], embedx: embeddings, path: str):
    return Chroma.from_documents(docx, embedx, persist_directory=path)


def make_retriever(store: vectorstore, search_type: str, kwargs: int):
    return store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": kwargs}
    )


# result = split_documents(load_document(RESUME_PDF_PATH))
# print(result[0].metadata)


# print(get_embeddings().embed_query("This is a test document")[:5])

# if __name__ == "__main__":
#     retriever = in_vector_store(
#         docx=split_documents(load_document(RESUME_PDF_PATH)),
#         embedx=get_embeddings(),
#         path=CHROMA_DB_PATH
#     ).as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 6}
#     )
#
#     retrieved_docx = retriever.invoke("PERSONAL DETAILS")
#
#     print(retrieved_docx[0].page_content)

if __name__ == "__main__":
    system_prompt_str = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved "
                         "context to answer the question. If you don't know the answer, say that you don't know. Use "
                         "three sentences maximum and keep the answer concise.\n\n{context}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt_str),
            ('human', '{input}')
        ]
    )

    # creating the vector store and using it

    # vector_store = to_vector_store(
    #     docx=load_document(RESUME_PDF_PATH),
    #     embedx=get_embeddings(),
    #     path=CHROMA_DB_PATH
    # )

    # using the already existing vector store

    vector_store = Chroma(
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DB_PATH
    )

    # retriever = make_retriever(vector_store, "mmr", 6)
    retriever = make_retriever(vector_store, "similarity", 6)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({
        "input": "What is your name and provide a formal introduction"
    })
    print(response["answer"])
