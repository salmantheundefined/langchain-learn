import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# --- STEP 1: State ---
class AgentState(TypedDict):
    question: str
    is_python: bool
    level: str       # "beginner" or "advanced"
    answer: str

# --- STEP 2: Nodes ---

def check_topic_node(state: AgentState):
    print("--- Checking topic ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a classifier. Your only job is to decide if a question is about Python programming.
Reply with ONLY the word 'yes' or 'no'. Nothing else. No punctuation.

Examples:
- 'What is a list in Python?' → yes
- 'How do decorators work?' → yes
- 'What is the capital of France?' → no
- 'How does GIL affect multithreading in Python?' → yes
- 'What is biryani?' → no"""),
        ("human", "Is this question about Python programming? Question: {question}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]})
    print(f"LLM raw response: '{result}'")
    is_python = result.strip().lower() == "yes"
    print(f"Is Python: {is_python}")
    return {"is_python": is_python}

# Node 2 — what level is the question?
def check_level_node(state: AgentState):
    print("--- Checking difficulty level ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Reply with only 'beginner' or 'advanced'. No other words."),
        ("human", "Is this a beginner or advanced Python question? Question: {question}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]})
    level = result.strip().lower()
    print(f"Level: {level}")
    return {"level": level}

# Node 3 — beginner answer
def beginner_node(state: AgentState):
    print("--- Answering as beginner ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python tutor for beginners. Use simple words, short sentences, and a basic example."),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer}

# Node 4 — advanced answer
def advanced_node(state: AgentState):
    print("--- Answering as advanced ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Python engineer. Give a detailed technical answer with advanced examples."),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer}

# Node 5 — decline non-Python
def decline_node(state: AgentState):
    print("--- Declining ---")
    return {"answer": "I only answer Python programming questions!"}

# --- STEP 3: Decision functions ---

def decide_topic(state: AgentState):
    if state["is_python"]:
        return "python"
    else:
        return "not_python"

def decide_level(state: AgentState):
    if state["level"] == "beginner":
        return "beginner"
    else:
        return "advanced"

# --- STEP 4: Build the graph ---
graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("check_topic", check_topic_node)
graph.add_node("check_level", check_level_node)
graph.add_node("beginner",    beginner_node)
graph.add_node("advanced",    advanced_node)
graph.add_node("decline",     decline_node)

# Entry point
graph.set_entry_point("check_topic")

# First conditional edge — Python or not?
graph.add_conditional_edges(
    "check_topic",
    decide_topic,
    {
        "python":     "check_level",
        "not_python": "decline"
    }
)

# Second conditional edge — beginner or advanced?
graph.add_conditional_edges(
    "check_level",
    decide_level,
    {
        "beginner": "beginner",
        "advanced": "advanced"
    }
)

# All end nodes go to END
graph.add_edge("beginner", END)
graph.add_edge("advanced", END)
graph.add_edge("decline",  END)

# --- STEP 5: Compile ---
app = graph.compile()

# --- STEP 6: Test ---
print("\n=== Test 1: Beginner Python question ===")
result = app.invoke({
    "question": "What is a variable in Python?",
    "is_python": False,
    "level": "",
    "answer": ""
})
print(f"\nAnswer: {result['answer']}")

print("\n=== Test 2: Advanced Python question ===")
result = app.invoke({
    "question": "How does Python's GIL affect multithreading performance?",
    "is_python": False,
    "level": "",
    "answer": ""
})
print(f"\nAnswer: {result['answer']}")

print("\n=== Test 3: Non-Python question ===")
result = app.invoke({
    "question": "What is the recipe for biryani?",
    "is_python": False,
    "level": "",
    "answer": ""
})
print(f"\nAnswer: {result['answer']}")