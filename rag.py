import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PERSIST_DIR = "chroma_db"

def load_and_split_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    # load pdf file
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()

    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)

    return chunks

def initialize_vector_store(chunks, embedding_model, collection_name="rag-chroma"):
    # Check if the vector database already exists
    if os.path.exists(PERSIST_DIR):
        # Load existing vector database from the specified directory
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(model=embedding_model, show_progress=True)
        )
    else:
        # Create a new vector database from documents and persist it
        vectordb = Chroma.from_documents(
            documents=chunks,
            collection_name=collection_name,
            embedding=OllamaEmbeddings(model=embedding_model, show_progress=True),
            persist_directory=PERSIST_DIR  # Persist directory for storage
        )
        vectordb.persist()  # Save the new vector database to disk
    return vectordb

def initialize_retriever(vectordb, llm):
    # retriever
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""As an AI assistant, your goal is to help the user find the most relevant information.
        Please rephrase the user's question in five different ways to capture various nuances and 
        intents. This will help in retrieving the most pertinent documents from the vector database. 
        Ensure each rephrased question is distinct and conveys the original meaning. 
        Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vectordb.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    return retriever

def setup_rag_chain(retriever, llm):
    after_rag_template = """Use the following pieces of context to answer the question at the end. 
    When you don't know the answer, say that you don't know, don't make things up.
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(after_rag_template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def main():
    # get data path
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    local_path = os.path.join(data_dir, 'time-management.pdf')
    embedding_model = 'nomic-embed-text'

    if not os.path.exists(PERSIST_DIR):
        print("Loading and splitting document...")
        chunks = load_and_split_document(file_path=local_path)
        vectordb = initialize_vector_store(chunks, embedding_model)
    else:
        print("Loading existing vector database...")
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name="rag-chroma",
            embedding_function=OllamaEmbeddings(model=embedding_model, show_progress=True)
        )

    # initialize model
    llm = ChatOllama(model="mistral")
    retriever = initialize_retriever(vectordb, llm)

    # setup RAG chain
    chain = setup_rag_chain(retriever, llm)
    

    while True:
        try:
            question = input("Ask me anything: ")
            if question.lower() == 'exit':
                print("Goodbye!")
                break

            response = chain.invoke(question)
            print("Response: ", response)
        except Exception as e:
            print("Error: ", e)

if __name__ == "__main__":
    main()