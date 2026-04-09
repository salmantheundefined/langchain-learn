import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

#INEDEXING THE RAG AFFILATED DOC
load_dotenv()

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
loader=TextLoader(os.path.join(BASE_DIR,"my_doc.txt"))
docs=loader.load()


splitter=RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=30)
chunks=splitter.split_documents(docs)

embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(chunks,embeddings)
retriever=vectorstore.as_retriever(search_kwargs={"k":2})
print("Document Indexed")


# --- CHAIN --- #
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt=ChatPromptTemplate([("system","""You are a helpful Python tutor.
Answer using ONLY the context below.
If not in context say 'I don't have that in my notes'.

Context: {context}"""),MessagesPlaceholder(variable_name='history'),("human","{question}")])

llm=ChatGroq(model="llama-3.3-70b-versatile")

chain=(
    {
        "context": RunnableLambda(lambda x:x["question"]) | retriever | format_docs,
        "question": RunnablePassthrough() | RunnableLambda(lambda x:x["question"]),
        "history": RunnablePassthrough() | RunnableLambda(lambda x:x["history"])
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --- CHAT LOOP -- #
history=[]
print("Convesational RAG CHATBOT - Type To exit")
print ("="*50)

while True:
    user_input=input("\n You:").strip()

    if user_input.lower()=="quit":
        print("Bye See you soon")
        break
    if not user_input:
        continue

    answer=chain.invoke({
        "question":user_input,
        "history": history
    })

    print(f"Meow_AI : {answer}")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))


