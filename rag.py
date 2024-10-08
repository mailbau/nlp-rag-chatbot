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

def load_and_split_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    # load pdf file
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()

    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)

    return chunks

def initialize_vector_store(chunks, embedding_model, collection_name="rag-chroma"):
    # convert chunks into embeddings and store in vector database
    vectordb = Chroma.from_documents(
        documents = chunks,
        collection_name = collection_name,
        embedding = OllamaEmbeddings(model=embedding_model, show_progress=True)
    )

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
    after_rag_template = """Answer the question based only on the following context:
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

    # load and split document
    chunks = load_and_split_document(file_path=local_path)
    vectordb = initialize_vector_store(chunks, embedding_model)

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