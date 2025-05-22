# faq_chatbot.py
import asyncio
import chromadb
import google.generativeai as genai
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

# Replace this with your Gemini API key
GOOGLE_API_KEY = "AIzaSyCHy80eWH_N7Q9Xc0niq9OpxdNKaCoJmBQ"

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Setup ChromaDB with Gemini embedding function
embedding_function = GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="faq", embedding_function=embedding_function)

# FAQ entries to be stored in ChromaDB
faq_data = {
    "What is RAG?": "Retrieval-Augmented Generation (RAG) combines retrieval with generation.",
    "How does ChromaDB work?": "ChromaDB is a vector DB used to store and search embeddings.",
    "What is Gemini API?": "Gemini is Google's advanced generative AI model."
}

# Add FAQs to ChromaDB only once
if collection.count() == 0:
    for question, answer in faq_data.items():
        collection.add(documents=[answer], ids=[question])

# Retriever class using ChromaDB
class RAGRetriever:
    def __init__(self, collection):
        self.collection = collection

    async def retrieve(self, query):
        results = self.collection.query(query_texts=[query], n_results=1)
        return results['documents'][0][0] if results['documents'] else "No info found."

# Handles the question and generates a response using Gemini
class QueryHandler:
    def __init__(self, model, retriever):
        self.model = model
        self.retriever = retriever

    async def handle_query(self, query):
        context = await self.retriever.retrieve(query)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = self.model.generate_content(prompt)  # ✅ No await here
        return response.text.strip()

# Chat system with round-robin logic (can support multiple agents)
class RoundRobinGroupChat:
    def __init__(self, agents):
        self.agents = agents
        self.index = 0

    async def ask(self, query):
        agent = self.agents[self.index]
        self.index = (self.index + 1) % len(self.agents)
        return await agent.handle_query(query)

# Main async loop
async def main():
    retriever = RAGRetriever(collection)
    query_handler = QueryHandler(model, retriever)
    chat = RoundRobinGroupChat([query_handler])

    print("📘 FAQ Chatbot (Gemini + ChromaDB). Type 'exit' to quit.")
    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            break
        answer = await chat.ask(question)
        print("🤖 Bot:", answer)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
