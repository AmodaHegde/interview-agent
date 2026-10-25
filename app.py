import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Interview RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("💡 Company Interview Question Assistant")
st.markdown("Ask questions about your custom interview question bank using Google Gemini with local document search.")

# ==========================================
# 2. CONFIGURATION & PATHS
# ==========================================
DATA_DIR = "./interview_docs"
VECTOR_DB_DIR = "./chroma_interview_db"
LLM_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "nomic-embed-text"

def get_google_api_key():
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# ==========================================
# 3. BACKEND INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def load_rag_system(api_key):
    # Ensure folder exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Ingest documents and build vector store if it doesn't exist
    if not os.path.exists(VECTOR_DB_DIR):
        loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".md": UnstructuredMarkdownLoader
        }
        all_docs = []
        for ext, loader_class in loaders.items():
            dir_loader = DirectoryLoader(
                DATA_DIR,
                glob=f"**/*{ext}",
                loader_cls=loader_class,
                loader_kwargs={"encoding": "utf-8"} if ext == ".txt" else {},
                show_progress=False
            )
            try:
                all_docs.extend(dir_loader.load())
            except Exception:
                pass

        if not all_docs:
            return None, "No documents found in './interview_docs'. Please add files and restart."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        splits = text_splitter.split_documents(all_docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
    else:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )

    # Set up retriever and chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=api_key)

    system_prompt = (
        "You are an expert technical interview coach assisting a candidate.\n"
        "Use the following retrieved context containing real company interview questions "
        "and guidance to answer the candidate's query accurately.\n"
        "If the answer cannot be determined from the context, state that clearly.\n\n"
        "Retrieved Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain, None

# Initialize RAG chain
google_api_key = get_google_api_key()
if not google_api_key:
    st.error("Google API key not found. Add GOOGLE_API_KEY=your_key to the project's .env file.")
    st.stop()

with st.spinner("Initializing local embeddings and document search..."):
    rag_chain, error_message = load_rag_system(google_api_key)

if error_message:
    st.error(error_message)
    st.info(f"👉 Drop your interview questions (.pdf, .txt, .csv, .docx) into the **`{DATA_DIR}`** folder and refresh the page.")
    st.stop()

# ==========================================
# 4. SIDEBAR & FILE MANAGEMENT
# ==========================================
with st.sidebar:
    st.subheader("📁 Document Management")
    st.markdown(f"Place your files inside:\n`{DATA_DIR}/`")
    
    if st.button("🔄 Re-index Database"):
        if os.path.exists(VECTOR_DB_DIR):
            import shutil
            shutil.rmtree(VECTOR_DB_DIR)
        st.cache_resource.clear()
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Config Info:**")
    st.write(f"- LLM: `{LLM_MODEL}`")
    st.write(f"- Embeddings: `{EMBEDDING_MODEL}`")
    st.write(f"- Vector DB: `ChromaDB`")

# ==========================================
# 5. CHAT INTERFACE & SESSION STATE
# ==========================================
# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input from chat box at the bottom
if user_query := st.chat_input("Ask a question about your interview question bank..."):
    # Append user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response using RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching questions and thinking..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
            except Exception as e:
                answer = f"An error occurred while generating the response: {e}"
            
            st.markdown(answer)
            
    # Append assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
