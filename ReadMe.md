# LangChain Learning Journey 
A structured, hands-on learning path covering LangChain, LangGraph, and Langfuse — built exercise by exercise with real working code.

---

## What This Is

This repo documents my learning journey through the LLM application development stack. Every concept is first understood, then implemented from scratch. No copy-pasting — every file was typed and debugged manually.

---

## Stack

- **LangChain** — chains, prompts, RAG pipelines
- **LangGraph** — agents and multi-step workflows *(in progress)*
- **Langfuse** — observability and tracing *(coming soon)*
- **Groq** — LLM provider (llama-3.3-70b-versatile)
- **FAISS** — local vector store
- **HuggingFace** — embeddings (all-MiniLM-L6-v2)

---

## Progress

| Stage | Topic | Status |
|-------|-------|--------|
| 1 | Foundations — LCEL, Prompts, Chains | ✅ Complete |
| 2 | RAG Pipeline — Load, Split, Embed, Retrieve | ✅ Complete |
| 3 | Memory — Multi-turn Conversation | ✅ Complete |
| 4 | LangGraph — Agents and Workflows | 🔄 In Progress |
| 5 | Langfuse — Observability and Tracing | ⬜ Coming Soon |

---

## Project Structure

```
langchain_learn/
│
├── lang_main/
│   ├── first_chain.py          # Stage 1 — first LCEL chain
│   ├── rag_chain.py            # Stage 2 — full RAG pipeline
│   ├── chatbot.py              # Stage 3 — multi-turn chatbot
│   ├── conversational_rag.py   # Stage 3 — RAG + memory combined
│   └── my_doc.txt              # sample document for RAG
│
├── .env                        # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/salmantheundefined/langchain-learn.git
cd langchain-learn
```

**2. Create virtual environment with Python 3.11**
```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
```

**3. Install dependencies**
```bash
pip install langchain langchain-groq langchain-community langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers python-dotenv
```

**4. Set up your API key**

Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## Files Explained

### `first_chain.py` — Stage 1
My first working LangChain chain. Demonstrates the core LCEL pipe operator connecting a prompt, LLM, and output parser.

```python
chain = prompt | llm | parser
result = chain.invoke({"question": "What is Python?"})
```

**Concepts covered:** PromptTemplate, ChatOpenAI, StrOutputParser, LCEL `|` pipe, `.invoke()` vs `.stream()` vs `.batch()`

---

### `rag_chain.py` — Stage 2
A full Retrieval-Augmented Generation pipeline. Loads a local text file, splits it into chunks, embeds them into a FAISS vector store, and answers questions grounded in the document.

```python
# Index once
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Query anytime
answer = rag_chain.invoke("What is a decorator?")
```

**Concepts covered:** TextLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, FAISS, similarity search, RAG chain

---

### `chatbot.py` — Stage 3
A multi-turn chatbot with conversation memory. Uses `MessagesPlaceholder` to inject history into every prompt so the LLM remembers previous turns.

```python
history = []
# Each turn appends HumanMessage + AIMessage to history
# History is passed into prompt via MessagesPlaceholder
```

**Concepts covered:** MessagesPlaceholder, HumanMessage, AIMessage, conversation history management

---

### `conversational_rag.py` — Stage 3
Combines RAG + memory into one chain. Answers questions from a document AND remembers the full conversation context.

```python
# Memory + RAG working together
You:     "What is a decorator?"
Meow_AI: "A decorator is a function that wraps another function..."

You:     "Can you remind me what I just asked about?"
Meow_AI: "You just asked about decorators."

You:     "What is machine learning?"
Meow_AI: "I don't have that in my notes."  ← grounded, not hallucinating
```

**Concepts covered:** Conversational RAG, RunnableLambda, RunnablePassthrough, combining retrieval with memory

---

## Key Learnings

**LCEL pipes everything together**
```python
chain = prompt | llm | parser   # declare the flow
chain.invoke({"question": "..."})  # run it
```

**RAG = open book exam for the LLM**
Instead of relying on training data, the LLM reads relevant chunks from your document before answering. No hallucinations on domain-specific content.

**Memory = a list of messages**
There's no magic memory object. It's just a Python list of `HumanMessage` and `AIMessage` objects that grows each turn and gets injected into the prompt via `MessagesPlaceholder`.

**Chunk size matters**
Too large → multiple topics in one chunk → imprecise retrieval
Too small → loses context at boundaries → incomplete answers
Sweet spot: 150–500 chars depending on document type

---

## Coming Next

- **Stage 4 — LangGraph:** Building agents that can reason, loop, and make decisions using `StateGraph`, nodes, edges, and conditional branching
- **Stage 5 — Langfuse:** Adding observability to trace every chain call, inspect retrieved chunks, and evaluate answer quality

---

## Resources

- [LangChain Docs](https://docs.langchain.com)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph)
- [Langfuse Docs](https://langfuse.com/docs)
- [Groq Console](https://console.groq.com)

---

*Learning in public — built step by step with a structured coaching approach.*