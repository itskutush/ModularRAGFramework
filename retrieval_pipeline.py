from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

#Step 4: Load embeddings and vector store 
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
    )

#Step 5: Search for relevant documents
query = "How did Microsoft establish itself as a tech giant?"

retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever = db.as_retriever(search_type="similarity_score_threshold",
#                              search_kwargs={"k": 5,
#                              "score_threshold": 0.3})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}\n")
print("--Context--")
for i,doc in enumerate(relevant_docs):
    print(f"Document {i}:\n {doc.page_content}")
