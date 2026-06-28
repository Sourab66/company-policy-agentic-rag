from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama

# Load env
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Company Policy RAG Assistant")

st.title("📘 Company Policy Assistant")

st.write("Ask questions from the employee handbook.")

# Load PDF
loader = PyPDFLoader("Data/Sample-Handbook.pdf")

documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# FREE local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# User question
question = st.text_input(
    "Ask a policy question"
)

if question:

    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Local AI model
    llm = ChatOllama(
        model="gemma3:1b"
    )

    prompt = f"""
You are a company HR policy assistant.

Answer ONLY from the provided handbook context.

If the answer is not found, say:
"I could not find that information in the handbook."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    # Output
    st.subheader("Answer")
    st.write(response.content)

    # Retrieved chunks
    st.subheader("Retrieved Context")

    for i, doc in enumerate(docs):
        st.write(f"### Chunk {i+1}")
        st.write(doc.page_content)