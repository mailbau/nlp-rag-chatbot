import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def main():
    # get data path
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    local_path = os.path.join(data_dir, 'time-management.pdf')

    # load pdf file
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()

    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    # convert chunks into embeddings and store in vector database
    vectordb = Chroma.from_documents(
        documents = chunks,
        collection_name = "rag-chroma",
        embedding = OllamaEmbeddings(model='nomic-embed-text', show_progress=True)
    )

    # Test retrieval with a sample query
    query = "What is time management?"
    results = vectordb.similarity_search(query, k=3)

    print("Top 3 most relevant chunks based on the query:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:\n{result.page_content}\n")


if __name__ == "__main__":
    main()