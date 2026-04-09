import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv
load_dotenv()

# 1. LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 2. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are {role} "),
    ("human", "{question}")
])

# 3. Parser
parser = StrOutputParser()

# 4. Chain
# chain = prompt | llm | parser

#creating partial chain
expert_chain=prompt.partial(role='python expert') | llm | parser
tutor_chain=prompt.partial(role='python tutor who explains simply') | llm | parser

# 5. Run
result1 = expert_chain.invoke({ "question": "what are args and kwargs"})
result2= tutor_chain.invoke({"question":"what are args and kwargs"})

print("=== Expert ===")
print(result1)
print("\n=== Tutor ===")
print(result2)