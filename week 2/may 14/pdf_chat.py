import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
from typing import List
from chromadb.api.types import EmbeddingFunction
from langchain.schema import Document
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain')

# === SETUP ===
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path_to_your_service_account.json'
genai.configure(api_key='your_gemini_api_key')

# === INIT CHROMA CLIENT ===
chroma_client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory='./chromadb_store'))

# Global collection (assigned later)
collection = None

# === FUNCTIONS ===
def get_pdf_text(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.3)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.embed_documents(input)

def store_chunks_in_chroma(chunks):
    global collection
    embedding_func = GoogleEmbeddingFunction()

    try:
        chroma_client.delete_collection('pdf_chunks')
    except:
        pass

    collection = chroma_client.create_collection(
        name='pdf_chunks',
        embedding_function=embedding_func
    )

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f'doc{i}'],
            metadatas=[{'source': f'chunk_{i}'}]
        )
    print('✅ Chunks stored in ChromaDB.')

def retrieve_relevant_chunks(question):
    results = collection.query(query_texts=[question], n_results=3)
    return [doc for doc in results['documents'][0] if doc]

def get_conversational_chain():
    prompt_template = '''
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "answer is not available in the context".

    Context: {context}
    Question: {question}

    Answer:
    '''
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return load_qa_chain(model, chain_type='stuff', prompt=prompt)

def ask_question(question):
    if collection is None:
        print('⚠️ Please load a PDF first.')
        return
    docs = retrieve_relevant_chunks(question)
    docs_for_chain = [Document(page_content=d) for d in docs]
    chain = get_conversational_chain()
    response = chain.run({'input_documents': docs_for_chain, 'question': question})
    print('\n💬 Answer:', response, '\n')