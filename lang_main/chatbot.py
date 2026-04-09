import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 2. Prompt with memory slot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Python tutor. Answer clearly and concisely."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 3. Chain
chain = prompt | llm | StrOutputParser()

# 4. History list — starts empty
history = []

print("Python Tutor Chatbot — type 'quit' to exit")
print("=" * 45)

# 5. Chat loop
while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "quit":
        print("Bye!")
        break

    if not user_input:
        continue

    # Invoke chain with history + question
    answer = chain.invoke({
        "history": history,
        "question": user_input
    })

    print(f"\nBot: {answer}")

    # Save this turn to history
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))