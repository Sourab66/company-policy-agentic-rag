from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama

# Load environment
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Agentic Company Policy Assistant")

st.title("🤖 Agentic Company Policy Assistant")

st.write("Ask questions from the employee handbook.")

# Load PDF
loader = PyPDFLoader("Data/Sample-Handbook.pdf")

documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 15
    }
)

# Local model
llm = ChatOllama(
    model="gemma3:4b"
)

# User question
question = st.text_input(
    "Ask a company policy question"
)

if question:

    # Agentic routing logic
    routing_prompt = f"""
You are an intelligent AI assistant.

Decide if the following question requires searching the company handbook.

Reply ONLY with:
YES
or
NO

Question:
{question}
"""

    route_decision = llm.invoke(
        routing_prompt
    ).content.strip()

    # Retrieval path
    if "YES" in route_decision:

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = f"""
You are a company HR policy assistant.

Answer ONLY using the provided handbook context.

If answer is not found, say:
"I could not find that information in the handbook."

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(final_prompt)

        st.subheader("Answer")

        st.write(response.content)

        st.subheader("Retrieved Context")

        for i, doc in enumerate(docs):

            st.write(f"### Chunk {i+1}")

            st.write(doc.page_content)

    # Non-retrieval path
    else:

        response = llm.invoke(question)

        st.subheader("Answer")

        st.write(response.content)