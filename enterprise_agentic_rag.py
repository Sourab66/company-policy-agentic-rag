"""
Enterprise Agentic Governance Engine (RAG)
============================================
Production-grade, self-correcting Retrieval-Augmented Generation system built on
LangGraph, ChromaDB, local Ollama (Gemma3), and CrossEncoder reranking.

This module preserves the original architecture (LangGraph workflow, MMR retrieval,
CrossEncoder reranking, LLM-as-a-Judge, query rewriting, retry loop) while adding:
    - True source attribution (page / chunk / document level)
    - Explainable retrieval (per-chunk scoring trace)
    - Retrieval diagnostics and observability (timing, counts)
    - A richer judge schema (faithfulness, relevance, confidence, grounded)
    - Governance guardrails that refuse low-confidence / ungrounded answers
    - A pre-generation retrieval-quality gate that triggers query rewriting
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

from langgraph.graph import StateGraph, END

# =============================================================================
# ENVIRONMENT INITIALIZATION
# =============================================================================
load_dotenv()

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
PDF_PATH: str = "Data/Sample-Handbook.pdf"
DOCUMENT_DISPLAY_NAME: str = "Employee Handbook"
PERSIST_DIRECTORY: str = "./chroma_db"

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 250

MMR_K: int = 8
MMR_FETCH_K: int = 25
RERANK_TOP_K: int = 4

# CrossEncoder (ms-marco-MiniLM) scores are unbounded logits; this threshold is a
# tunable heuristic for "at least one chunk is plausibly relevant".
RERANK_SCORE_THRESHOLD: float = -2.0

# Governance thresholds
CONFIDENCE_THRESHOLD: float = 0.75
MAX_RETRIES: int = 2

FALLBACK_NOT_FOUND: str = "I could not find this information in the handbook."
FALLBACK_LOW_CONFIDENCE: str = (
    "I could not confidently answer this question using the available handbook information."
)

# =============================================================================
# WEB INTERFACE CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Enterprise AI Governance Engine", layout="wide")
st.title("Enterprise AI Agentic RAG")
st.markdown("### Production-Grade Self-Correcting Architecture with Automated Validation Loops")

# =============================================================================
# CORE MODEL DEPENDENCIES
# =============================================================================
@st.cache_resource
def load_cross_encoder() -> CrossEncoder:
    """Load and cache the CrossEncoder reranking model."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def load_llm() -> ChatOllama:
    """Load and cache the local Ollama chat model."""
    return ChatOllama(model="gemma3:4b")


reranker = load_cross_encoder()
llm = load_llm()


# =============================================================================
# LIGATURE & CHARACTER SANITIZATION
# =============================================================================
def sanitize_handbook_text(text: str) -> str:
    """Fix typographic font / encoding corruptions found in the handbook PDF.

    Args:
        text: Raw extracted PDF text, possibly containing ligature artifacts.

    Returns:
        Cleaned text with whitespace normalized and ligatures restored.
    """
    text = re.sub(r"\s+", " ", text)
    ligature_map = {
        "ě": "ff",  # Oěences -> Offences
        "Ĵ": "tt",  # commiĴed -> committed
        "Ĝ": "fi",  # suĜciently -> sufficiently
        "Ě": "fl",  # Ěexibility -> flexibility
    }
    for corrupt_char, clean_char in ligature_map.items():
        text = text.replace(corrupt_char, clean_char)
    return text.strip()


# =============================================================================
# PRODUCTION VECTOR COMPUTE LAYER
# =============================================================================
@st.cache_resource
def load_vectorstore() -> Chroma:
    """Load, sanitize, chunk, embed, and persist the handbook into ChromaDB.

    Each chunk is annotated with provenance metadata (document name, 1-indexed
    page number, and a globally unique chunk id) so that downstream nodes can
    perform true source attribution instead of guessing.

    Returns:
        A populated Chroma vector store ready for retrieval.
    """
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    for doc in docs:
        doc.page_content = sanitize_handbook_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # Enrich every chunk with deterministic, non-hallucinated provenance metadata.
    for chunk_id, chunk in enumerate(chunks):
        raw_page = chunk.metadata.get("page", 0)
        chunk.metadata["document_name"] = DOCUMENT_DISPLAY_NAME
        chunk.metadata["page_number"] = int(raw_page) + 1  # PyPDFLoader is 0-indexed
        chunk.metadata["chunk_id"] = chunk_id

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K},
)


# =============================================================================
# STATE SCHEMA (LangGraph Engine)
# =============================================================================
class RetrievedChunk(TypedDict):
    """Explainable retrieval record for a single chunk."""

    chunk_id: int
    page_number: int
    document_name: str
    score: float
    content: str


class Timings(TypedDict):
    """Cumulative wall-clock timings across the workflow, in seconds."""

    embedding_search_time: float
    reranking_time: float
    llm_generation_time: float
    total_workflow_time: float


class AgentState(TypedDict, total=False):
    """Full shared state propagated through the LangGraph workflow."""

    question: str
    current_search_query: str
    search_history: List[str]

    docs: List[Document]
    retrieved_chunks: List[RetrievedChunk]
    retrieved_pages: List[int]
    retrieved_sources: List[str]
    retrieval_scores: List[float]
    needs_rewrite: bool

    context: str
    answer: str

    faithfulness_score: float
    relevance_score: float
    confidence: float
    grounded: bool
    missing_information: bool
    used_pages: List[int]
    eval_score: float
    eval_rationale: str

    retry_count: int
    diagnostics: dict
    timings: Timings


# =============================================================================
# DEDUPLICATION & RERANKING UTILITIES
# =============================================================================
def deduplicate(docs: List[Document]) -> List[Document]:
    """Remove exact-duplicate chunks (e.g. from MMR overlap) by normalized text.

    Args:
        docs: Candidate documents returned by the retriever.

    Returns:
        Documents with duplicate text content removed, order preserved.
    """
    seen = set()
    out: List[Document] = []
    for d in docs:
        text = " ".join(d.page_content.split())
        if text not in seen:
            seen.add(text)
            out.append(d)
    return out


def rerank(
    query: str,
    docs: List[Document],
    top_k: int = RERANK_TOP_K,
    min_score: Optional[float] = None,
) -> List[RetrievedChunk]:
    """Score documents with the CrossEncoder and return an explainable trace.

    Always returns the top-k highest-scoring chunks if any documents were
    retrieved — it never silently discards all evidence. Quality gating
    (e.g. deciding whether to trigger a query rewrite) is handled separately
    by validate_retrieval_node, not by dropping chunks here.

    Args:
        query: The search query used to score relevance.
        docs: Deduplicated candidate documents.
        top_k: Number of top-scoring chunks to keep.
        min_score: Optional score floor. If provided, chunks below it are
            excluded UNLESS doing so would leave zero chunks, in which case
            the floor is ignored so at least one chunk is always returned.

    Returns:
        A list of RetrievedChunk records (doc + score + provenance), sorted by
        descending CrossEncoder score, truncated to top_k.
    """
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    if min_score is not None:
        filtered = [(doc, score) for doc, score in ranked if score >= min_score]
        # Never let filtering zero out retrieval evidence entirely — fall back
        # to the unfiltered ranking so downstream nodes still get the best
        # available chunks, even if none clear the bar.
        scored = (filtered or ranked)[:top_k]
    else:
        scored = ranked[:top_k]

    trace: List[RetrievedChunk] = []
    for doc, score in scored:
        trace.append(
            RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id", -1),
                page_number=doc.metadata.get("page_number", -1),
                document_name=doc.metadata.get("document_name", DOCUMENT_DISPLAY_NAME),
                score=float(score),
                content=doc.page_content,
            )
        )
    return trace


# =============================================================================
# CITATION / FORMATTING HELPERS
# =============================================================================
def compress_page_ranges(pages: List[int]) -> str:
    """Compress a list of page numbers into human-readable ranges.

    Example: [13, 14, 15, 17] -> "13-15, 17"

    Args:
        pages: List of (possibly unsorted, possibly duplicate) page numbers.

    Returns:
        A comma-separated, range-compressed string.
    """
    unique_sorted = sorted(set(p for p in pages if p and p > 0))
    if not unique_sorted:
        return "N/A"

    ranges: List[str] = []
    start = prev = unique_sorted[0]
    for page in unique_sorted[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(ranges)


# =============================================================================
# GRAPH NODE 1: Smart Retrieval
# =============================================================================
def retrieve_node(state: AgentState) -> dict:
    """Retrieve candidate chunks via MMR, deduplicate, and rerank with CrossEncoder.

    Populates explainable retrieval metadata (scores, pages, sources) and tracks
    embedding-search / reranking timing for observability.
    """
    search_query = state.get("current_search_query") or state["question"]
    retry_count = state.get("retry_count", 0)
    timings = dict(state.get("timings") or _empty_timings())

    embed_start = time.perf_counter()
    raw_docs = retriever.invoke(search_query)
    timings["embedding_search_time"] += time.perf_counter() - embed_start

    deduped = deduplicate(raw_docs)

    rerank_start = time.perf_counter()
    retrieved_chunks = rerank(search_query, deduped, top_k=RERANK_TOP_K)
    timings["reranking_time"] += time.perf_counter() - rerank_start

    diagnostics = dict(state.get("diagnostics") or {})
    diagnostics.update(
        {
            "mmr_retrieval_count": len(raw_docs),
            "chunks_after_dedup": len(deduped),
            "chunks_after_rerank": len(retrieved_chunks),
        }
    )

    return {
        "docs": [Document(page_content=c["content"]) for c in retrieved_chunks],
        "retrieved_chunks": retrieved_chunks,
        "retrieved_pages": [c["page_number"] for c in retrieved_chunks],
        "retrieved_sources": sorted({c["document_name"] for c in retrieved_chunks}),
        "retrieval_scores": [c["score"] for c in retrieved_chunks],
        "current_search_query": search_query,
        "retry_count": retry_count,
        "diagnostics": diagnostics,
        "timings": timings,
    }


def _empty_timings() -> Timings:
    """Return a zeroed timings dict."""
    return Timings(
        embedding_search_time=0.0,
        reranking_time=0.0,
        llm_generation_time=0.0,
        total_workflow_time=0.0,
    )


# =============================================================================
# GRAPH NODE 2: Retrieval Quality Gate
# =============================================================================
def validate_retrieval_node(state: AgentState) -> dict:
    """Check whether retrieval found at least one plausibly-relevant chunk.

    If no chunk clears RERANK_SCORE_THRESHOLD and retries remain, flags the
    state so the router sends the question back through query rewriting
    instead of generating an answer from weak context.
    """
    scores = state.get("retrieval_scores", [])
    best_score = max(scores) if scores else float("-inf")
    needs_rewrite = best_score < RERANK_SCORE_THRESHOLD
    return {"needs_rewrite": needs_rewrite}


def retrieval_quality_router(state: AgentState) -> str:
    """Route to query rewriting if retrieval quality is weak and retries remain."""
    if state.get("needs_rewrite") and state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    return "proceed"


# =============================================================================
# GRAPH NODE 3: Context Construction
# =============================================================================
def context_node(state: AgentState) -> dict:
    """Build a structured, attribution-rich context block for generation."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"context": "No relevant context was retrieved."}

    sections = []
    for chunk in chunks:
        sections.append(
            "Source:\n{doc}\n\nPage:\n{page}\n\nChunk:\n{chunk_id}\n\nContent:\n{content}".format(
                doc=chunk["document_name"],
                page=chunk["page_number"],
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
            )
        )
    return {"context": "\n\n---\n\n".join(sections)}


# =============================================================================
# GRAPH NODE 4: Protected Generation
# =============================================================================
def answer_node(state: AgentState) -> dict:
    """Generate a grounded, cited answer constrained strictly to retrieved context."""
    question = state["question"]
    context = state["context"]
    timings = dict(state.get("timings") or _empty_timings())

    prompt = f"""You are a strict Corporate HR Policy Assistant.

INSTRUCTIONS:
1. Answer using ONLY the information in the context below. Never invent or assume policies.
2. Be concise and directly address the question — no filler.
3. Ground every claim in the context and cite the page it came from inline, in the
   format (Employee Handbook, p.<page>) or (Employee Handbook, pp.<start>-<end>) for
   claims spanning multiple pages. Only cite pages that actually appear in the context.
4. If the context is incomplete or only partially answers the question, give the
   partial answer and explicitly state what is uncertain or missing.
5. If the context does NOT contain the answer at all, reply ONLY with:
   "{FALLBACK_NOT_FOUND}"
6. Do NOT combine a successful answer with the fallback phrase — use one or the other.

Context:
{context}

Question:
{question}

Answer:"""

    gen_start = time.perf_counter()
    response = llm.invoke(prompt).content
    timings["llm_generation_time"] += time.perf_counter() - gen_start

    return {"answer": response, "timings": timings}


# =============================================================================
# GRAPH NODE 5: LLM-as-a-Judge Auditor
# =============================================================================
def validation_judge_node(state: AgentState) -> dict:
    """Score the generated answer for faithfulness, relevance, and grounding."""
    context = state["context"]
    answer = state["answer"]
    timings = dict(state.get("timings") or _empty_timings())

    judge_prompt = f"""You are an elite Quality Assurance AI evaluating HR system answers against source documents.
Evaluate the answer based ONLY on the provided context.

Context:
{context}

System Answer:
{answer}

CRITERIA RULES:
1. faithfulness_score: 1.0 if every claim is supported by the context, 0.0 if it hallucinates.
2. relevance_score: does the answer directly address the question?
   CRITICAL PENALTY: if the answer contains "{FALLBACK_NOT_FOUND}", relevance_score MUST be 0.0.
3. confidence: your overall confidence (0.0-1.0) that this answer is correct and complete.
4. grounded: true only if every claim traces back to the context, else false.
5. missing_information: true if the context only partially covers the question.
6. used_pages: list the integer page numbers (from the context) that the answer actually relies on.

Respond STRICTLY as raw JSON inside a markdown code block, nothing else:
{{
    "faithfulness_score": 1.0,
    "relevance_score": 0.0,
    "confidence": 0.0,
    "grounded": true,
    "missing_information": false,
    "rationale": "Brief explanation.",
    "used_pages": []
}}"""

    gen_start = time.perf_counter()
    response = llm.invoke(judge_prompt).content
    timings["llm_generation_time"] += time.perf_counter() - gen_start

    metrics = _parse_judge_response(response)

    eval_score = (metrics["faithfulness_score"] + metrics["relevance_score"]) / 2.0

    return {
        "faithfulness_score": metrics["faithfulness_score"],
        "relevance_score": metrics["relevance_score"],
        "confidence": metrics["confidence"],
        "grounded": metrics["grounded"],
        "missing_information": metrics["missing_information"],
        "used_pages": metrics["used_pages"],
        "eval_score": eval_score,
        "eval_rationale": metrics["rationale"],
        "timings": timings,
    }


def _parse_judge_response(response: str) -> dict:
    """Safely parse the judge's JSON output, with a graceful fail-safe default.

    Handles missing keys, malformed JSON, and non-JSON responses without ever
    raising — the caller always receives a complete, well-typed metrics dict.
    """
    defaults = {
        "faithfulness_score": 0.0,
        "relevance_score": 0.0,
        "confidence": 0.0,
        "grounded": False,
        "missing_information": True,
        "rationale": "JSON parsing failure during automated quality analysis.",
        "used_pages": [],
    }
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return defaults
        parsed = json.loads(match.group(0))
        return {
            "faithfulness_score": float(parsed.get("faithfulness_score", 0.0)),
            "relevance_score": float(parsed.get("relevance_score", 0.0)),
            "confidence": float(parsed.get("confidence", 0.0)),
            "grounded": bool(parsed.get("grounded", False)),
            "missing_information": bool(parsed.get("missing_information", True)),
            "rationale": str(parsed.get("rationale", "")) or defaults["rationale"],
            "used_pages": [int(p) for p in parsed.get("used_pages", []) if isinstance(p, (int, float))],
        }
    except Exception:
        return defaults


# =============================================================================
# GRAPH NODE 6: Dynamic Query Rewriter
# =============================================================================
def fallback_query_rewrite_node(state: AgentState) -> dict:
    """Rewrite the search query when retrieval or generation quality is weak."""
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    search_history = list(state.get("search_history") or [question])

    rewrite_prompt = f"""Transform this user policy question into clean semantic keywords optimized for a database search index.
Remove all section numbers, punctuation, or complex logic phrases.

Original Question: {question}
Optimized Database Search Keywords:"""

    new_query = llm.invoke(rewrite_prompt).content.strip()
    search_history.append(new_query)

    return {
        "current_search_query": new_query,
        "retry_count": retry_count,
        "search_history": search_history,
        "needs_rewrite": False,
    }


# =============================================================================
# GRAPH NODE 7: Governance Finalization
# =============================================================================
def finalize_node(state: AgentState) -> dict:
    """Apply governance guardrails before the answer reaches the user.

    Refuses to surface answers that fall below the confidence threshold or are
    flagged as ungrounded by the judge, regardless of how the retry loop ended.
    """
    confidence = state.get("confidence", 0.0)
    grounded = state.get("grounded", False)
    answer = state.get("answer", "")

    if confidence < CONFIDENCE_THRESHOLD or not grounded:
        if answer != FALLBACK_NOT_FOUND:
            answer = FALLBACK_LOW_CONFIDENCE
    return {"answer": answer}


# =============================================================================
# CONDITIONAL ROUTING CONTROLLER
# =============================================================================
def corrective_router(state: AgentState) -> str:
    """Evaluate pipeline trajectory using automated compliance scoring metrics."""
    if state.get("eval_score", 0.0) >= CONFIDENCE_THRESHOLD:
        return "pass_to_ui"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "max_loops_exceeded"
    return "recompute_and_retry"


# =============================================================================
# GRAPH ORCHESTRATION BUILDER
# =============================================================================
def build_graph() -> StateGraph:
    """Assemble the LangGraph workflow with retrieval gating and governance."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("validate_retrieval", validate_retrieval_node)
    graph.add_node("context", context_node)
    graph.add_node("answer", answer_node)
    graph.add_node("judge", validation_judge_node)
    graph.add_node("rewrite", fallback_query_rewrite_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "validate_retrieval")
    graph.add_conditional_edges(
        "validate_retrieval",
        retrieval_quality_router,
        {"rewrite": "rewrite", "proceed": "context"},
    )
    graph.add_edge("context", "answer")
    graph.add_edge("answer", "judge")
    graph.add_conditional_edges(
        "judge",
        corrective_router,
        {
            "pass_to_ui": "finalize",
            "max_loops_exceeded": "finalize",
            "recompute_and_retry": "rewrite",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("finalize", END)

    return graph


app_graph = build_graph().compile()


# =============================================================================
# STREAMLIT RUNTIME UI INTERACTION
# =============================================================================
def render_confidence_dashboard(result: AgentState) -> None:
    """Render the confidence / observability metrics panel."""
    st.subheader("📊 Confidence Dashboard")

    timings = result.get("timings", _empty_timings())

    row1 = st.columns(4)
    row1[0].metric("Faithfulness", f"{result.get('faithfulness_score', 0.0) * 100:.1f}%")
    row1[1].metric("Relevance", f"{result.get('relevance_score', 0.0) * 100:.1f}%")
    row1[2].metric("Overall Confidence", f"{result.get('confidence', 0.0) * 100:.1f}%")
    row1[3].metric("Retry Count", result.get("retry_count", 0))

    row2 = st.columns(4)
    row2[0].metric("Retrieved Pages", len(set(result.get("retrieved_pages", []))))
    row2[1].metric("Retrieved Chunks", len(result.get("retrieved_chunks", [])))
    row2[2].metric("Source Documents", len(result.get("retrieved_sources", [])))
    row2[3].metric("Grounded", "Yes" if result.get("grounded") else "No")

    st.markdown("**⏱️ Observability — Timing Breakdown**")
    row3 = st.columns(4)
    row3[0].metric("Embedding Search", f"{timings.get('embedding_search_time', 0.0):.2f}s")
    row3[1].metric("CrossEncoder Rerank", f"{timings.get('reranking_time', 0.0):.2f}s")
    row3[2].metric("LLM Generation", f"{timings.get('llm_generation_time', 0.0):.2f}s")
    row3[3].metric("Total Workflow", f"{timings.get('total_workflow_time', 0.0):.2f}s")

    st.info(f"**Auditor Rationale:** {result.get('eval_rationale', 'N/A')}")


def render_sources_used(result: AgentState) -> None:
    """Render the 'Sources Used' summary, computed from retrieval metadata only."""
    pages = result.get("retrieved_pages", [])
    sources = result.get("retrieved_sources", [DOCUMENT_DISPLAY_NAME])

    st.markdown("**Sources Used**")
    for source in sources:
        st.write(source)
    st.write(f"Pages: {compress_page_ranges(pages)}")


def render_retrieved_sources_expander(result: AgentState) -> None:
    """Expander: which documents/pages were retrieved for this question."""
    with st.expander("📚 Retrieved Sources"):
        chunks = result.get("retrieved_chunks", [])
        if not chunks:
            st.write("No sources retrieved.")
            return
        for chunk in chunks:
            st.write(f"- **{chunk['document_name']}** — Page {chunk['page_number']} (Chunk {chunk['chunk_id']})")


def render_retrieval_trace_expander(result: AgentState) -> None:
    """Expander: per-chunk explainability trace with CrossEncoder scores."""
    with st.expander("🔍 Retrieval Trace"):
        chunks = result.get("retrieved_chunks", [])
        if not chunks:
            st.write("No retrieval trace available.")
            return
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {i}**")
            st.write(f"Page: {chunk['page_number']}")
            st.write(f"CrossEncoder Score: {chunk['score']:.4f}")
            preview = chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else "")
            st.caption(f"Chunk Preview: {preview}")
            st.divider()


def render_validation_report_expander(result: AgentState) -> None:
    """Expander: full LLM-as-a-Judge governance report."""
    with st.expander("✅ Validation Report"):
        st.json(
            {
                "faithfulness_score": result.get("faithfulness_score", 0.0),
                "relevance_score": result.get("relevance_score", 0.0),
                "confidence": result.get("confidence", 0.0),
                "grounded": result.get("grounded", False),
                "missing_information": result.get("missing_information", False),
                "used_pages": result.get("used_pages", []),
                "rationale": result.get("eval_rationale", ""),
                "retry_count": result.get("retry_count", 0),
            }
        )


def render_retrieval_diagnostics_expander(result: AgentState) -> None:
    """Expander: raw retrieval pipeline transparency (counts, queries)."""
    with st.expander("🛠️ Retrieval Diagnostics"):
        diagnostics = result.get("diagnostics", {})
        st.write(f"Original Question: {result.get('question', '')}")
        st.write(f"Search Query: {result.get('current_search_query', '')}")
        st.write(f"Search History: {result.get('search_history', [])}")
        st.write(f"MMR Retrieval Count: {diagnostics.get('mmr_retrieval_count', 0)}")
        st.write(f"Chunks after Deduplication: {diagnostics.get('chunks_after_dedup', 0)}")
        st.write(f"Chunks after Reranking: {diagnostics.get('chunks_after_rerank', 0)}")
        st.write(f"Retry Count: {result.get('retry_count', 0)}")


def run_app() -> None:
    """Render the Streamlit UI and drive the LangGraph workflow on submit."""
    col1, _col2 = st.columns([2, 1])
    with col1:
        question = st.text_input("📝 Ask corporate handbook policy question:")

    if not question:
        return

    with st.spinner("Processing through LangGraph Validation Engine..."):
        workflow_start = time.perf_counter()
        result = app_graph.invoke(
            {
                "question": question,
                "retry_count": 0,
                "search_history": [question],
                "timings": _empty_timings(),
            }
        )
        result["timings"]["total_workflow_time"] = time.perf_counter() - workflow_start

    st.subheader("📌 Verified System Answer")
    st.write(result.get("answer", FALLBACK_NOT_FOUND))
    render_sources_used(result)

    st.divider()
    render_confidence_dashboard(result)

    st.divider()
    render_retrieval_diagnostics_expander(result)
    render_retrieved_sources_expander(result)
    render_retrieval_trace_expander(result)
    render_validation_report_expander(result)


if __name__ == "__main__":
    run_app()