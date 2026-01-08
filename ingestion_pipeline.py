'''Ingestion pipeline'''
import os 
from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


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


def main():
    print("Main Function")

    #1. Load Documents
    documents = load_documentts(docs_path="docs")

    #2. chunk files
    #3. embedding and storing vector db

    
if __name__ == "__main__":
    main()