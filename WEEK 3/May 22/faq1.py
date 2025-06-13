import asyncio
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from autogen import ConversableAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage
import os
import logging

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('faq1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Gemini API Setup ===
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error("Failed to configure Gemini API: " + str(e))
    raise

async def gemini_generate(prompt: str, model_name="gemini-1.5-flash", retries: int = 5, delay: int = 60) -> str:
    model = genai.GenerativeModel(model_name)
    for i in range(retries):
        try:
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(prompt))
            logger.info("Gemini API call successful on attempt " + str(i + 1))
            return response.text
        except Exception as e:
            logger.warning(f"Gemini error. Retrying in {delay}s ({i+1}/{retries}): {str(e)}")
            await asyncio.sleep(delay)
    logger.error("Gemini API failed after retries")
    return "ERROR: Gemini API failed after retries."

class GeminiModelClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

    async def create(self, messages, **kwargs):
        try:
            prompt = "\n".join([msg["content"] for msg in messages])
            response = await gemini_generate(prompt, model_name=self.model_name)
            return {"choices": [{"message": {"content": response}}]}
        except Exception as e:
            return {"choices": [{"message": {"content": "ERROR: Failed to generate response - " + str(e)}}]}

# === PDF Loading and Chunking ===
def load_faq_text(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    logger.info("Loaded PDF: " + pdf_path + ", pages: " + str(len(texts)))
    return texts

def split_chunks(texts: list, chunk_size: int = 300, overlap: int = 50) -> list:
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    logger.info("Created " + str(len(chunks)) + " chunks")
    return chunks

def setup_chromadb(chunks: list) -> chromadb.Collection:
    embedding_func = GoogleGenerativeAiEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        model_name="models/embedding-001"
    )
    client = chromadb.Client()
    collection = client.get_or_create_collection("faq_chunks", embedding_function=embedding_func)
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            metadatas=[{"source": f"chunk_{i}"}]
        )
    logger.info("ChromaDB collection created with " + str(len(chunks)) + " chunks")
    return collection

# === Custom Agents ===
class RAGRetrieverAgent(ConversableAgent):
    produced_message_types = [TextMessage]

    def __init__(self, collection: chromadb.Collection, model_client: GeminiModelClient, **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.model_client = model_client
        logger.info("Initialized RAGRetrieverAgent")

    async def a_generate_reply(self, messages, sender, config=None):
        try:
            latest_query = messages[-1]["content"]
            results = self.collection.query(query_texts=[latest_query], n_results=5)

            if not results["documents"] or not results["documents"][0]:
                return True, "I couldn't find any relevant information in the knowledge base."

            context = "\n\n".join(results["documents"][0])
            prompt = f"""You are an AI assistant that answers based only on the context below.

User Question: {latest_query}

Relevant Information:
{context}

Respond in bullet points or numbered steps.
If no relevant answer is found, say so clearly.
"""
            response = await gemini_generate(prompt, model_name=self.model_client.model_name)
            return True, response
        except Exception as e:
            return True, "ERROR: Failed to process query - " + str(e)

class UserProxyAgentWithProduced(UserProxyAgent):
    produced_message_types = [TextMessage]

# === Main Runner ===
async def main():
    faq_path = r'C:\Users\sahee\OneDrive\Desktop\Lealabs\WEEK 3\May 22\FAQs.pdf'
    try:
        texts = load_faq_text(faq_path)
        chunks = split_chunks(texts)
        collection = setup_chromadb(chunks)
    except Exception as e:
        print("❌ Failed to process PDF:", str(e))
        return

    model_client = GeminiModelClient(api_key=GOOGLE_API_KEY)

    user_proxy = UserProxyAgentWithProduced(
        name="User",
        human_input_mode="ALWAYS",
        code_execution_config=False
    )

    retriever_agent = RAGRetrieverAgent(
        name="RAGRetriever",
        collection=collection,
        model_client=model_client,
        llm_config=False,
        system_message="You retrieve and answer questions using a RAG system."
    )

    groupchat = SelectorGroupChat(
        [user_proxy, retriever_agent],
        model_client=model_client,
        selector_prompt="Select the next agent to respond based on the last message. The RAGRetriever should answer queries, and the User can ask questions or terminate with 'exit'.",
        termination_condition=lambda msg: "exit" in msg.get("content", "").lower(),
        allow_repeated_speaker=True
    )

    print("🚀 Starting FAQ pipeline. Type your question (or 'exit' to quit):")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            print("✅ Exiting FAQ pipeline.")
            break

        try:
            reply = await groupchat.a_initiate_chat(
                message=TextMessage(content=question, source="User")
            )
            print("Answer: " + reply.content)
        except Exception as e:
            print("❌ Failed to get answer:", str(e))


if __name__ == "__main__":
    asyncio.run(main())
