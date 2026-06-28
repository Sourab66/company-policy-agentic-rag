from dotenv import load_dotenv
import streamlit as st
import re
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from sentence_transformers import CrossEncoder

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

# -----------------------------
# ENVIRONMENT INITIALIZATION
# -----------------------------
load_dotenv()

# -----------------------------
# WEB INTERFACE CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Enterprise AI Governance Engine", layout="wide")
st.title("Enterprise Agentic Governance Engine (RAG)")
st.markdown("### Production-Grade Self-Correcting Architecture with Automated Validation Loops")

# -----------------------------
# CORE MODEL DEPENDENCIES
# -----------------------------
@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_cross_encoder()
llm = ChatOllama(model="gemma3:4b")

# -----------------------------
# LIGATURE & CHARACTER SANITIZATION 
# -----------------------------
def sanitize_handbook_text(text: str) -> str:
    """Fixes typographic font and encoding corruptions found in the handbook PDF."""
    text = re.sub(r'\s+', ' ', text)
    ligature_map = {
        "ě": "ff",      # Oěences -> Offences
        "Ĵ": "tt",      # commiĴed -> committed
        "Ĝ": "fi",      # suĜciently -> sufficiently
        "Ě": "fl",      # Ěexibility -> flexibility
    }
    for corrupt_char, clean_char in ligature_map.items():
        text = text.replace(corrupt_char, clean_char)
    return text.strip()

# -----------------------------
# PRODUCTION VECTOR COMPUTE LAYER
# -----------------------------
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("Data/Sample-Handbook.pdf")
    docs = loader.load()

    # Pre-clean document contents to fix corrupted vector storage/matching
    for doc in docs:
        doc.page_content = sanitize_handbook_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250
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
    search_kwargs={"k": 8, "fetch_k": 25}
)

# -----------------------------
# ENHANCED STATE SCHEMA (LangGraph Engine)
# -----------------------------
class AgentState(TypedDict):
    question: str
    current_search_query: str
    docs: list
    context: str
    answer: str
    eval_score: float
    eval_rationale: str
    retry_count: int

# -----------------------------
# DEDUPLICATION & RERANKING UTILITIES
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

def rerank(query, docs, top_k=3):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

# -----------------------------
# GRAPH NODE 1: Smart Retrieval
# -----------------------------
def retrieve_node(state: AgentState):
    # Fallback to base question if current_search_query is not initialized
    search_query = state.get("current_search_query") or state["question"]
    retry_count = state.get("retry_count", 0)
    
    docs = retriever.invoke(search_query)
    docs = deduplicate(docs)
    docs = rerank(search_query, docs, top_k=4)

    return {"docs": docs, "current_search_query": search_query, "retry_count": retry_count}

# -----------------------------
# GRAPH NODE 2: Context Construction
# -----------------------------
def context_node(state: AgentState):
    docs = state["docs"]
    context = "\n\n".join(f"[Source Chunk {i+1}]: {d.page_content}" for i, d in enumerate(docs))
    return {"context": context}

# -----------------------------
# GRAPH NODE 3: Protected Generation (UPGRADED)
# -----------------------------
def answer_node(state: AgentState):
    question = state["question"]
    context = state["context"]

    prompt = f"""You are a strict Corporate HR Policy Assistant.

INSTRUCTIONS:
1. Provide a direct, complete answer using ONLY the context below.
2. Ground every single claim specifically to the context text provided.
3. If the context does NOT contain the answer, reply ONLY with: "I could not find this information in the handbook."
4. Do NOT combine a successful answer with the fallback phrase. If you answer the question, do not include the fallback sentence.

Context:
{context}

Question:
{question}

Answer:"""

    response = llm.invoke(prompt).content
    return {"answer": response}

# -----------------------------
# GRAPH NODE 4: LLM-as-a-Judge Auditor
# -----------------------------
def validation_judge_node(state: AgentState):
    """Automated evaluation node inspecting answer grounding and hallucination metrics."""
    context = state["context"]
    answer = state["answer"]
    
    judge_prompt = f"""You are an elite Quality Assurance AI evaluating HR system answers against source documents.
Evaluate the correctness of the answer based ONLY on the provided context.

Context:
{context}

System Answer:
{answer}

CRITERIA RULES:
1. Faithfulness: Does the answer contain claims or guesses not written in the context? Score 1.0 if faithful, 0.0 if it hallucinates.
2. Answer Relevance: Does the answer directly solve and answer the user question? 
   - CRITICAL PENALTY: If the system answer contains the phrase "I could not find this information in the handbook", you MUST score relevance_score as 0.0. A fallback message is not a valid answer.

Respond STRICTLY in a raw JSON schema format inside a markdown code block, containing nothing else:
{{
    "faithfulness_score": 1.0,
    "relevance_score": 0.0,
    "rationale": "Write a brief explanation sentence here."
}}"""

    response = llm.invoke(judge_prompt).content
    
    # Secure JSON extractor block parsing
    try:
        clean_json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        metrics = json.loads(clean_json_str)
        score = (float(metrics["faithfulness_score"]) + float(metrics["relevance_score"])) / 2.0
        rationale = metrics["rationale"]
    except Exception:
        # Strict fail-safe default if local LLM fails to output parseable schema
        score = 0.0
        rationale = "JSON parsing failure during automated quality analysis orchestration."

    return {"eval_score": score, "eval_rationale": rationale}


# -----------------------------
# GRAPH NODE 5: Dynamic Query Rewriter
# -----------------------------
def fallback_query_rewrite_node(state: AgentState):
    """Transforms search terms if previous context results produced unfaithful generation."""
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    
    rewrite_prompt = f"""Transform this user policy question into clean semantic keywords optimized for a database search index.
Remove all section numbers, punctuation, or complex logic phrases.

Original Question: {question}
Optimized Database Search Keywords:"""

    new_query = llm.invoke(rewrite_prompt).content.strip()
    return {"current_search_query": new_query, "retry_count": retry_count}

# -----------------------------
# CONDITIONAL ROUTING CONTROLLER
# -----------------------------
def corrective_router(state: AgentState) -> str:
    """Evaluates pipeline trajectory using automated compliance scoring metrics."""
    # Step out if execution passes accuracy bars
    if state.get("eval_score", 0.0) >= 0.75:
        return "pass_to_ui"
    
    # Cap loop iterations at 2 passes max to contain latency and token inflation
    if state.get("retry_count", 0) >= 2:
        return "max_loops_exceeded"
        
    return "recompute_and_retry"

# -----------------------------
# GRAPH ORCHESTRATION BUILDER
# -----------------------------
graph = StateGraph(AgentState)

# Append functional engine nodes
graph.add_node("retrieve", retrieve_node)
graph.add_node("context", context_node)
graph.add_node("answer", answer_node)
graph.add_node("judge", validation_judge_node)
graph.add_node("rewrite", fallback_query_rewrite_node)

# Set workflow pipeline edges
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "context")
graph.add_edge("context", "answer")
graph.add_edge("answer", "judge")

# Conditional loop edge logic
graph.add_conditional_edges(
    "judge",
    corrective_router,
    {
        "pass_to_ui": END,
        "max_loops_exceeded": END,
        "recompute_and_retry": "rewrite"
    }
)
graph.add_edge("rewrite", "retrieve")

app_graph = graph.compile()

# -----------------------------
# STREAMLIT RUNTIME UI INTERACTION
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input("📝 Ask corporate handbook policy question:")

if question:
    with st.spinner("Processing through LangGraph Validation Engine..."):
        result = app_graph.invoke({
            "question": question,
            "retry_count": 0
        })

    # Render verified system response
    st.subheader("📌 Verified System Answer")
    st.write(result["answer"])
    
    # Diagnostic metric block interface visualization
    with col2:
        st.subheader("📊 Engine Execution Metrics")
        st.metric(label="Validation Confidence Score", value=f"{result.get('eval_score', 0.0) * 100:.1f}%")
        st.info(f"**Auditor Rationale:** {result.get('eval_rationale', 'N/A')}")
        st.metric(label="Workflow Engine Routing Passes", value=f"{result.get('retry_count', 0) + 1}")

    # Trace telemetry expansion container
