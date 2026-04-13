from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import *

load_dotenv()

llm=ChatGroq(model='llama-3.3-70b-versatile')

class AgentState(TypedDict):
    question : str
    is_python : bool
    answer : str

def check_node(state : AgentState):
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([(
        "system","reply with only yes or no "
    ),
        (
            "human", f"you should only answer the question which is related to python {question}"
        )])

    parser=StrOutputParser()

    chain = prompt | llm | parser

    result = chain.invoke({"question":question})

    is_python = result.strip().lower() == 'yes'

    print(f"is_python = {is_python}")

    return {"is_python":is_python}

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

def decision_path(state : AgentState):
    if state["is_python"]:
        return "python"
    else :
        return "not_python"

graph=StateGraph(AgentState)

graph.add_node("check_topic",check_node)
graph.add_node("answer",answer_node)
graph.add_node("decline",decline_node)

graph.set_entry_point("check_topic")

graph.add_conditional_edges(
    "check_topic",
    decision_path,
    {
        "python":"answer",
        "not_python":"decline"
    }
)

graph.add_edge("answer",END)
graph.add_edge("decline", END)


app=graph.compile()

print("testing for python question by keeping the is_python false")
result = app.invoke(
    {
        "question":"what is use of kwargs?",
        "is_python":False,
        "answer":""
    }
)

print(f"answer : {result['answer']}")

print("testing for non python question by keeping the is_python true")
result2=app.invoke(
    {
        "question":"what is the capital of india",
        "is_python":True,
        "answer":""
    }
)

print(f"answer : {result['answer']}")




