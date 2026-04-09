import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

#phase 1
#loading the text documents using textloader

loader=TextLoader(r"C:\Users\Salman\OneDrive\Desktop\langchain_learn\lang_main\my_doc.txt")
docs=loader.load()
print(f"Loaded {len(docs)} document")
print(f"Characters in doc: {len(docs[0].page_content)}")
print(f"Preview: '{docs[0].page_content[:300]}'")

#split the text into chunks RecursiveCharacter Text splitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)
chunks=splitter.split_documents(docs)
print(f"split into {len(docs)} into chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: '{chunk.page_content[:80]}'")

#embeded and store
embeddings=HuggingFaceEmbeddings(
    model="all-MiniLM-L6-v2"
)
vectorstore=FAISS.from_documents(chunks,embeddings)
print("created vectorstore succesfully")

#step4 to create a retreiver to match the closest vectors
retreiver= vectorstore.as_retriever(search_kwargs={"k":2})


prompt=ChatPromptTemplate.from_messages([
    ("system","""Answer using ONLY the context below.
If the answer is not in the context, say 'I don't know'.

Context:
{context}"""),
    ("human", "{question}")
])

llm=ChatGroq(model="llama-3.3-70b-versatile")

#format retreived docs into strings
def format_strings(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain=(
    {"context": retreiver | format_strings, "question": RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)

#ask a question:
question="What is machine learning"
print(f"question is {question}")

answer=rag_chain.invoke(question)
print(f"answer is {answer}")

# --- INSPECT RETRIEVED CHUNKS ---
print("\n" + "=" * 50)
print("INSPECTING RETRIEVED CHUNKS")
print("=" * 50)

test_questions = [
    "What is a decorator?",
    "How do I add items to a list?",
    "What are kwargs?",
    "What is machine learning?",  # not in doc
]

for question in test_questions:
    print(f"\nQuestion: {question}")
    print("-" * 30)

    # Get chunks directly from retriever
    chunks = retreiver.invoke(question)

    print(f"Retrieved {len(chunks)} chunk(s):")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}: '{chunk.page_content}'")

    # Get the answer
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")












