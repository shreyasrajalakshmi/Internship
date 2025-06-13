import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
from typing import List
from chromadb.api.types import EmbeddingFunction
from langchain.schema import Document
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='langchain')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')

# === SET CREDENTIALS DIRECTLY ===
# Service account JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\sahee\OneDrive\Desktop\Lealabs\WEEK 2\May 14\expanded-symbol-459706-c2-1300aa0869c4.json"

# API key (used by generativeai SDK)
genai.configure(api_key="AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")

# === INIT CHROMA CLIENT ===
chroma_client = chromadb.PersistentClient(path="chromadb_store")

# Global collection (assigned later)
collection = None

# === FUNCTIONS ===

def get_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)





class GoogleEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.embed_documents(input)

def store_chunks_in_chroma(chunks):
    global collection

    embedding_func = GoogleEmbeddingFunction()

    try:
        chroma_client.delete_collection("pdf_chunks")
    except:
        pass

    collection = chroma_client.create_collection(
        name="pdf_chunks",
        embedding_function=embedding_func
    )

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"doc{i}"],
            metadatas=[{"source": f"chunk_{i}"}]
        )
    print("✅ Chunks stored in ChromaDB.")



def retrieve_relevant_chunks(question):
    results = collection.query(query_texts=[question], n_results=3)
    return [doc for doc in results["documents"][0] if doc]

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "answer is not available in the context".

    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def ask_question(question):
    if collection is None:
        print("⚠️ Please load a PDF first.")
        return
    docs = retrieve_relevant_chunks(question)
    docs_for_chain = [Document(page_content=d) for d in docs]  # ✅ Use actual Document objects
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs_for_chain, "question": question})
    print("\n💬 Answer:", response["output_text"], "\n")


# === MAIN MENU ===

def main():
    while True:
        print("\n==== PDF Chatbot CLI ====")
        print("1. Load PDF")
        print("2. Ask a question")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            path = input("Enter path to PDF file: ").strip()
            if os.path.exists(path):
                print("📄 Extracting text...")
                text = get_pdf_text(path)
                print("✂️ Chunking text...")
                chunks = get_text_chunks(text)
                print("💾 Storing in ChromaDB...")
                store_chunks_in_chroma(chunks)
                print("✅ PDF processed and ready to chat.")
            else:
                print("❌ Invalid file path. Please try again.")

        elif choice == "2":
            question = input("Ask your question: ")
            ask_question(question)

        elif choice == "3":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
