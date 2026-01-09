from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


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
query = "How much Microsoft pay to acquire Github?" #replace with any synthetic query

retriever = db.as_retriever(search_kwargs={"k": 3})

retriever = db.as_retriever(search_type="similarity_score_threshold",
                             search_kwargs={"k": 5,
                             "score_threshold": 0.3})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}\n")
print("--Context--")
for i,doc in enumerate(relevant_docs):
    print(f"Document {i}:\n {doc.page_content}\n")


# Step 6: Generate answer using LLM based on the user query

combineed_input = f"""Use the below context to answer the question.{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. if you can't find the answer then just repsond with "I don't know"."""

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2"
)
hf_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"
)

transformer_pipeline = pipeline(
    "text-generation",
    model=hf_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.3,
)

lc_pipeline = HuggingFacePipeline(pipeline=transformer_pipeline)
model = ChatHuggingFace(llm=lc_pipeline)



messages = [
    SystemMessage(content="You are a helpful assistant that helps users find information."),
    HumanMessage(content=combineed_input),  
    ]

result = model.invoke(messages)

print("\n--generated answer--")

print("content only:")
print(result.content)
      
