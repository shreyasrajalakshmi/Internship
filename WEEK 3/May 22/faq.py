import asyncio
import fitz  # PyMuPDF
import os
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from google.generativeai import GenerativeModel
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage
import google.generativeai as genai
# Hardcoded API key (replace with your actual key)
GOOGLE_API_KEY = "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg"
genai.configure(api_key=GOOGLE_API_KEY)
# ---------------------------
# Load and preprocess the FAQ PDF
# ---------------------------
def load_faq_text(pdf_path):
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    return texts

def split_chunks(texts, chunk_size=300, overlap=50):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

# ---------------------------
# Set up ChromaDB with Gemini embeddings
# ---------------------------
def setup_chromadb(chunks):
    embedding_func = GoogleGenerativeAiEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        model_name="models/embedding-001"
    )

    client = chromadb.Client()
    collection = client.get_or_create_collection("faq_chunks", embedding_function=embedding_func)

    documents = [{"id": str(i), "documents": chunk, "metadatas": {"source": f"chunk_{i}"}} for i, chunk in enumerate(chunks)]
    for doc in documents:
        collection.add(
            ids=[doc["id"]],
            documents=[doc["documents"]],
            metadatas=[doc["metadatas"]]
        )

    return collection

# ---------------------------
# Async RAG Retriever agent
# ---------------------------
class RAGRetrieverAgent(ConversableAgent):
    def __init__(self, collection, **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")

    async def a_generate_reply(self, messages, sender, config=None):
        latest_query = messages[-1]["content"]

        results = self.collection.query(
            query_texts=[latest_query],
            n_results=5
        )

        if not results["documents"][0]:
            return False, "I couldn't find any relevant information in the knowledge base."

        context = "\n\n".join(results["documents"][0])

        prompt = f"""You are an AI assistant that answers based only on the context below.

User Question: {latest_query}

Relevant Information:
{context}

Respond in bullet points or numbered steps.
If no relevant answer is found, say so clearly.
"""

        response = self.gemini.generate_content(prompt)
        return True, response.text

# ---------------------------
# User-proxy and chat manager
# ---------------------------
async def main():
    faq_path = r'C:\Users\sahee\OneDrive\Desktop\Lealabs\WEEK 3\May 22\FAQs.pdf'
    texts = load_faq_text(faq_path)
    chunks = split_chunks(texts)
    collection = setup_chromadb(chunks)

    user_proxy = UserProxyAgent(name="User", code_execution_config=False)

    retriever_agent = RAGRetrieverAgent(
        name="RAGRetriever",
        collection=collection,
        llm_config=False,
        system_message="You retrieve and answer questions using a RAG system."
    )

    groupchat = SelectorGroupChat(
        [user_proxy, retriever_agent],
        model_client=model_client,
        selector_prompt=retriever_agent,
        
        
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=False)

    user_question = "How do I keep track of my returned orders?"
    await user_proxy.a_initiate_chat(
        recipient=manager,
        message=TextMessage(content=user_question)
    )

if __name__ == "__main__":
    asyncio.run(main())
