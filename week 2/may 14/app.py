import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and persist vector store using Chroma
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()
    return vectorstore

# Function to create a prompt-based QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, just say "Answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user queries
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    relevant_docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    answer = chain.run(input_documents=relevant_docs, question=user_question)
    return answer

# --- Streamlit UI ---
st.set_page_config(page_title="📄 PDF Chatbot with Gemini", layout="wide")

# Sidebar: Upload & Process PDFs
with st.sidebar:
    st.title("📄 PDF Upload")
    st.markdown("Upload your **PDF documents** and click **Process** to extract and index content.")
    pdf_docs = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("🔍 Process PDFs"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        st.success("✅ PDFs processed successfully!")

# Main UI
# Improved output box styling
st.markdown(
    """
    <style>
        .answer-container {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-top: 1rem;
            font-size: 1.1rem;
            line-height: 1.6;
            color: #333333;
        }
        .question-label {
            font-size: 1.2rem;
            font-weight: 600;
            color: #444444;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Chat with your PDF using Gemini + ChromaDB")

with st.container():
    st.subheader("Ask questions based on your uploaded PDFs")
    user_question = st.text_input("🔎 Type your question here")

    if user_question:
        with st.spinner("🔍 Thinking..."):
            response = user_input(user_question)
            st.markdown("### 🧠 Answer:")
            st.markdown(f"<div class='question-box'>{response}</div>", unsafe_allow_html=True)