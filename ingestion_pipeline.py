'''Ingestion pipeline'''
import os 
from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

#Step 1: Load Documents
def load_documentts(docs_path = "docs"):
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    for i , doc in enumerate(documents[:2]):
        print(f"Document {i+1} :")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...\n")
        print(f" metadata: {doc.metadata}\n")
        
    return documents

# Step 2: Chunk files
def split_documents(documents,chunk_size=1000, chunk_overlap=0):

    print("splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    if chunks:
        for i,chunk in enumerate(chunks[:5]):
            print(f"Chunk {i+1} :")
            print(f" Source: {chunk.metadata['source']}")
            print(f" Length: {len(chunk.page_content)}charecters")
            print(f" Content: ")
            print(chunk.page_content)
            print("-*"*20)

        if len(chunks) > 5:
               print(f"... {len(chunks) - 5} more chunks")
    return chunks

# Step 3: Embedding and storing in vector DB
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating vector embeddings and storing them in chroma DB...")

    embedding_model = HuggingFaceEmbeddings(model_name = "intfloat/e5-base-v2")

    print("---Creating a vector store--")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata = {"hnsw:space":"cosine"}
    )
    print("Finished creating a vector store")
    
    print(f"Vector store persisted at {persist_directory}")

    return vector_store









def main():
    print("Main Function")

    #1. Load Documents
    documents = load_documentts(docs_path="docs")

    #2. chunk files

    chunks = split_documents(documents)
    #3. embedding and storing vector db
    vectorstore = create_vector_store(chunks)
    
if __name__ == "__main__":
    main()