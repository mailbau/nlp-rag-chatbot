import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    # Test if the document is split properly by printing the chunk details
    print(f"Total number of chunks: {len(chunks)}\n")
    
    # Print the content of the first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i + 1}:\n{chunk.page_content}\n{'-'*40}\n")

if __name__ == "__main__":
    main()