from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate(
    [("system","you are an useful assistant"),
     ("human", "{question}")]
)

parser=StrOutputParser()

chain = prompt | llm | parser

result=chain.invoke({"question":"explain the complexity behind coding in simple words"})

print("="*25)
print(result)
print("="*25)
