from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from sentence_transformers import CrossEncoder

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# -----------------------------
# ENV
# -----------------------------
load_dotenv()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="LangGraph RAG Agent")
st.title("LangGraph Production RAG Agent")

# -----------------------------
# Reranker
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# LLM
# -----------------------------
llm = ChatOllama(model="gemma3:4b")

# -----------------------------
# Vector DB
# -----------------------------
@st.cache_resource
def load_vectorstore():

    loader = PyPDFLoader("Data/Sample-Handbook.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
)

# -----------------------------
# STATE (LangGraph memory)
# -----------------------------
class AgentState(TypedDict):
    question: str
    docs: list
    context: str
    answer: str

# -----------------------------
# HELPERS
# -----------------------------
def deduplicate(docs):
    seen = set()
    out = []

    for d in docs:
        text = " ".join(d.page_content.split())
        if text not in seen:
            seen.add(text)
            out.append(d)

    return out


def rerank(question, docs, top_k=3):
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

# -----------------------------
# NODE 1: Retrieve
# -----------------------------
def retrieve_node(state: AgentState):
    question = state["question"]

    docs = retriever.invoke(question)
    docs = deduplicate(docs)
    docs = rerank(question, docs, top_k=3)

    return {"docs": docs}

# -----------------------------
# NODE 2: Build Context
# -----------------------------
def context_node(state: AgentState):
    docs = state["docs"]

    context = "\n\n".join(d.page_content for d in docs)

    return {"context": context}

# -----------------------------
# NODE 3: Generate Answer
# -----------------------------
def answer_node(state: AgentState):
    question = state["question"]
    context = state["context"]

    prompt = f"""
You are a strict HR assistant.

RULES:
- Use ONLY the context below
- If answer not found, say:
  "I could not find this information in the handbook."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt).content

    return {"answer": response}

# -----------------------------
# BUILD GRAPH
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("context", context_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "context")
graph.add_edge("context", "answer")
graph.add_edge("answer", END)

app_graph = graph.compile()

# -----------------------------
# STREAMLIT INPUT
# -----------------------------
question = st.text_input("Ask company policy question")

if question:

    with st.spinner("Running LangGraph agent..."):

        result = app_graph.invoke({
            "question": question
        })

        st.subheader("📌 Answer")
        st.write(result["answer"])

        st.success("Powered by LangGraph RAG Agent")

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(result["docs"]):
                st.markdown(f"### Chunk {i+1}")
                st.write(doc.page_content)