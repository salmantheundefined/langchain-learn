import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain.agents import AgentState
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=ChatGroq(model='llama-3.3-70b-versatile')

#building a State Notepad

class AgentState(TypedDict):
    question: str
    is_python : bool
    answer : str

#stage-2 building nodes

def check_topic_node(state: AgentState):
    print("----Checking the Topic-----")

    question=state["question"]

    prompt=ChatPromptTemplate([("system","reply with 'yes' or 'no' no other options "),
                               ("human","is this question about python programming? Question: {question}")])

    chain=prompt|llm|StrOutputParser()

    result=chain.invoke({"question":question})

    is_python= result.strip().lower()=="yes"
    print(f"is python : {is_python}")
    return {"is_python": is_python}

def answer_node(state: AgentState):
    print("---Answering the question---")

    question=state["question"]

    prompt=ChatPromptTemplate([("system","you are useful python assistant tutor answer it clearly"),("human","{question}")])

    parser=StrOutputParser()

    chain=prompt|llm|parser

    answer=chain.invoke({"question":question})

    return {"answer":answer}

def decline_node(state : AgentState):
    print("---declining its a non-python question---")

    return {"answer":"i only answer something in python can you please ask something in python"}


#decision function
def decide_path(state:AgentState):
    if state["is_python"]:
        return "python"
    else:
        return "not_python"

#building the graph

graph=StateGraph(AgentState)

#adding nodes
graph.add_node("check_topic",check_topic_node)
graph.add_node("answer",answer_node)
graph.add_node("decline",decline_node)

#setting up the starting node
graph.set_entry_point("check_topic")

#add conditional edges from check_topic

graph.add_conditional_edges(
    "check_topic",
    decide_path,
    {
        "python": "answer",
        "not_python": "decline"
    }
)

graph.add_edge("answer",END)
graph.add_edge("decline",END)

app=graph.compile()

print("\n=== Test 1: Python question ===")
result = app.invoke({
    "question": "What is a list comprehension in Python?",
    "is_python": False,
    "answer": ""
})
print(f"Answer: {result['answer']}")

print("\n=== Test 2: Non-Python question ===")
result = app.invoke({
    "question": "What is the capital of France?",
    "is_python": False,
    "answer": ""
})
print(f"Answer: {result['answer']}")



