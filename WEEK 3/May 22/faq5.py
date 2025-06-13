import asyncio
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from autogen import ConversableAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage
import os
import logging
import re

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('faq5.log')  # Log to file only
    ]
)
logger = logging.getLogger(__name__)

# === Gemini API Setup ===
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error("Failed to configure Gemini API: %s", e)
    raise

async def gemini_generate(prompt: str, model_name="gemini-1.5-flash", retries: int = 5, delay: int = 60) -> str:
    """Generate content using Gemini API with retry logic."""
    model = genai.GenerativeModel(model_name)
    for i in range(retries):
        try:
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(prompt))
            logger.info("Gemini API call successful on attempt %d", i + 1)
            return response.text
        except genai.types.generation_types.BlockedPromptException as e:
            logger.error("Gemini API blocked prompt: %s", e)
            return "ERROR: Prompt blocked by Gemini API."
        except genai.types.generation_types.StopCandidateException as e:
            logger.error("Gemini API stopped candidate: %s", e)
            return "ERROR: Response generation stopped by Gemini API."
        except Exception as e:
            if "429" in str(e):
                logger.error("Gemini API quota exceeded: %s", e)
                return "ERROR: Gemini API quota exceeded. Please check your plan at https://ai.google.dev/gemini-api/docs/rate-limits."
            logger.warning("Gemini error. Retrying in %ds (%d/%d): %s", delay, i + 1, retries, e)
            await asyncio.sleep(delay)
    logger.error("Gemini API failed after %d retries", retries)
    return "ERROR: Gemini API failed after retries."

# === Custom Gemini ModelClient Wrapper ===
class GeminiModelClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

    async def create(self, messages, **kwargs):
        try:
            prompt = "\n".join([msg["content"] for msg in messages])
            response = await gemini_generate(prompt, model_name=self.model_name)
            logger.info("GeminiModelClient created response")
            return {"choices": [{"message": {"content": response}}]}
        except Exception as e:
            logger.error("GeminiModelClient failed: %s", e)
            return {"choices": [{"message": {"content": "ERROR: Failed to generate response - %s" % e}}]}

# === PDF Loading and Chunking ===
def load_faq_text(pdf_path: str) -> list:
    try:
        if not os.path.exists(pdf_path):
            logger.error("PDF file not found: %s", pdf_path)
            raise FileNotFoundError("PDF file not found: %s" % pdf_path)
        doc = fitz.open(pdf_path)
        texts = [page.get_text() for page in doc]
        logger.info("Loaded PDF: %s, pages: %d", pdf_path, len(texts))
        return texts
    except Exception as e:
        logger.error("Failed to load PDF: %s", e)
        raise

def split_chunks(texts: list, chunk_size: int = 500, overlap: int = 100) -> list:
    """Split text into chunks, attempting to split by FAQ entries."""
    chunks = []
    for text in texts:
        # Try to split by FAQ entries (e.g., lines starting with numbers or questions)
        faq_entries = re.split(r'(\d+\.\s+.*?|\bHow\b.*?\?)', text, flags=re.IGNORECASE)
        faq_entries = [entry.strip() for entry in faq_entries if entry.strip()]
        
        current_chunk = ""
        current_length = 0
        
        for entry in faq_entries:
            entry_words = entry.split()
            entry_length = len(entry_words)
            
            if current_length + entry_length <= chunk_size:
                current_chunk += entry + "\n"
                current_length += entry_length
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = entry + "\n"
                current_length = entry_length
                
                # Handle overlap
                if chunks and overlap > 0:
                    prev_chunk = chunks[-1]
                    overlap_words = prev_chunk.split()[-overlap:]
                    current_chunk = " ".join(overlap_words) + " " + current_chunk
                    current_length += len(overlap_words)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    logger.info("Created %d chunks", len(chunks))
    return chunks

# === ChromaDB Setup with Gemini Embeddings ===
def setup_chromadb(chunks: list) -> chromadb.Collection:
    try:
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
        logger.info("ChromaDB collection created with %d chunks", len(chunks))
        return collection
    except Exception as e:
        logger.error("Failed to setup ChromaDB: %s", e)
        raise

# === Query Handler Agent ===
class QueryHandlerAgent(ConversableAgent):
    produced_message_types = [TextMessage]

    def __init__(self, model_client: GeminiModelClient, **kwargs):
        super().__init__(**kwargs)
        self.model_client = model_client
        logger.info("Initialized QueryHandlerAgent")

    async def a_generate_reply(self, messages=None, sender=None, config=None):
        try:
            logger.info("QueryHandlerAgent a_generate_reply called with messages=%s, sender=%s", messages, sender)
            if not messages:
                logger.warning("No messages provided to QueryHandlerAgent")
                return True, TextMessage(content="ERROR: No query provided. Please ask a question.", source="QueryHandler")

            latest_query = messages[-1]["content"].strip()
            logger.info("QueryHandlerAgent processing query: %s", latest_query)

            if not latest_query:
                logger.warning("Empty query received")
                return True, TextMessage(content="ERROR: Please provide a valid question.", source="QueryHandler")

            prompt = f"""You are a query handler for an FAQ chatbot. Validate and rephrase the user's query to make it clear, concise, and aligned with common FAQ phrasing in a document. Accept short or fragmented questions and rephrase them into full questions. Preserve key terms (e.g., '2FA', 'return', 'window period') and include synonyms (e.g., 'enable', 'configure', 'return policy', 'two-factor authentication') to improve retrieval. Return only the rephrased query.

User Query: {latest_query}

Instructions:
- If clear, return unchanged or slightly refined.
- If short or fragmented (e.g., 'return window period?'), rephrase into a full question (e.g., 'What is the return window period?').
- If vague, rephrase for specificity, keeping original terms and adding synonyms.
- If invalid (e.g., offensive, gibberish), return "INVALID: Please ask a valid FAQ-related question."
- Do not answer the query; only rephrase.
- Examples:
  - 'how to set up 2FA?' -> 'How to enable two-factor authentication (2FA)?'
  - 'return window period?' -> 'What is the return window period?'
  - '2FA setup?' -> 'How to enable two-factor authentication (2FA)?'
"""
            rephrased_query = await gemini_generate(prompt, model_name=self.model_client.model_name)
            logger.info("Rephrased query: %s", rephrased_query)
            return True, TextMessage(content=rephrased_query, source="QueryHandler")
        except Exception as e:
            logger.error("QueryHandlerAgent failed: %s", e)
            return True, TextMessage(content="ERROR: Failed to process query - %s" % e, source="QueryHandler")

# === RAG Retriever Agent ===
class RAGRetrieverAgent(ConversableAgent):
    produced_message_types = [TextMessage]

    def __init__(self, collection: chromadb.Collection, model_client: GeminiModelClient, **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.model_client = model_client
        logger.info("Initialized RAGRetrieverAgent")

    async def a_generate_reply(self, messages=None, sender=None, config=None):
        try:
            logger.info("RAGRetrieverAgent a_generate_reply called with messages=%s, sender=%s", messages, sender)
            if not messages:
                logger.warning("No messages provided to RAGRetrieverAgent")
                return True, TextMessage(content="ERROR: No query provided for retrieval.", source="RAGRetriever")

            latest_query = messages[-1]["content"].strip()
            logger.info("RAGRetrieverAgent processing query: %s", latest_query)

            # Query expansion: generate variants to improve retrieval
            query_variants = [latest_query]
            if "2FA" in latest_query or "two-factor" in latest_query.lower():
                query_variants.extend([
                    latest_query.replace("2FA", "two-factor authentication"),
                    latest_query.replace("set up", "enable"),
                    latest_query.replace("set up", "configure")
                ])
            if "return" in latest_query.lower():
                query_variants.extend([
                    latest_query.replace("return window", "return policy"),
                    latest_query.replace("period", "timeframe")
                ])
            query_variants = list(set(query_variants))  # Remove duplicates
            logger.info("Query variants: %s", query_variants)

            # Query ChromaDB with all variants
            all_results = []
            for query in query_variants:
                results = self.collection.query(query_texts=[query], n_results=10)
                if results["documents"] and results["documents"][0]:
                    all_results.extend(results["documents"][0])
                    logger.info("Retrieved %d chunks for query: %s", len(results["documents"][0]), query)

            if not all_results:
                logger.warning("No relevant documents found for query: %s", latest_query)
                return True, TextMessage(content="I couldn't find any relevant information in the knowledge base.", source="RAGRetriever")

            # Deduplicate and limit context
            context = "\n\n".join(list(dict.fromkeys(all_results))[:5])
            logger.info("Retrieved context: %s...", context[:200])

            prompt = f"""You are an AI assistant that answers based only on the context below.

User Question: {latest_query}

Relevant Information:
{context}

Respond in bullet points or numbered steps.
If no relevant answer is found, say so clearly.
"""
            response = await gemini_generate(prompt, model_name=self.model_client.model_name)
            logger.info("Generated response: %s...", response[:200])
            return True, TextMessage(content=response, source="RAGRetriever")
        except Exception as e:
            logger.error("RAGRetrieverAgent failed: %s", e)
            return True, TextMessage(content="ERROR: Failed to process query - %s" % e, source="RAGRetriever")

# === Custom SelectorGroupChat to Fix Issues ===
class CustomSelectorGroupChat(SelectorGroupChat):
    def __init__(self, agents, model_client, selector_prompt, termination_condition, allow_repeated_speaker):
        super().__init__(agents, model_client, selector_prompt=selector_prompt, termination_condition=termination_condition, allow_repeated_speaker=allow_repeated_speaker)
        self.agents = agents
        self.max_round = 10
        self.messages = []
        self.model_client = model_client
        self.selector_prompt = selector_prompt
        self.query_handler_count = 0  # Track QueryHandler iterations
        logger.info("Initialized CustomSelectorGroupChat with max_round=%d, messages=%d, selector_prompt=%s...", 
                    self.max_round, len(self.messages), selector_prompt[:50])

    def append(self, message, *args):
        """Append a message to the chat history, ensuring valid format."""
        try:
            if isinstance(message, TextMessage):
                message = {"content": message.content, "role": "user", "source": message.source}
            if not isinstance(message, dict):
                message = {"content": str(message), "role": "user", "source": "Unknown"}
            if "content" not in message:
                message["content"] = "ERROR: Invalid message format"
            if "role" not in message:
                message["role"] = "user"
            if "source" not in message:
                message["source"] = "Unknown"
            self.messages.append(message)
            logger.info("Appended message to CustomSelectorGroupChat: %s..., extra args: %s", str(message)[:200], args)
        except Exception as e:
            logger.error("Failed to append message: %s", e)
            self.messages.append({"content": "ERROR: Failed to append message - %s" % e, "role": "assistant", "source": "System"})

    async def a_select_speaker(self, last_speaker, selector):
        """Select the next speaker based on the selector prompt using Gemini."""
        try:
            logger.info("a_select_speaker called with last_speaker=%s, selector=%s", last_speaker, selector)
            last_message = self.messages[-1] if self.messages else {}
            last_content = last_message.get("content", "")
            last_source = last_message.get("source", "")
            logger.info("Selecting speaker after last message: %s (source: %s)", last_content[:200], last_source)

            # Force RAGRetriever if QueryHandler has run once
            if self.query_handler_count >= 1:
                logger.info("QueryHandler already run, forcing RAGRetriever")
                for agent in self.agents:
                    if agent.name == "RAGRetriever":
                        return agent
                logger.warning("RAGRetriever not found, defaulting to QueryHandler")
                return self.agents[0]

            prompt = f"""{self.selector_prompt}

Last Message: {last_content}
Message Source: {last_source}

Available Agents: QueryHandler, RAGRetriever

Instructions:
- If the last message is from User and is a question, select QueryHandler to validate/rephrase.
- If the last message is from QueryHandler and is a valid question (not starting with 'INVALID' or 'ERROR'), select RAGRetriever to answer.
- If the last message contains 'exit', select None to terminate.
- If the last message is invalid or an error, select None.
- Return only the agent name (QueryHandler, RAGRetriever) or 'None'.
"""
            response = await gemini_generate(prompt, model_name=self.model_client.model_name)
            response = response.strip()
            logger.info("Speaker selection response: '%s'", response)
            logger.info("Available agent names: %s", [agent.name for agent in self.agents])

            if response == "None" or "exit" in last_content.lower() or "INVALID" in last_content or "ERROR" in last_content:
                logger.info("Termination condition met, no speaker selected")
                return None

            for agent in self.agents:
                if agent.name.lower() == response.lower():
                    if agent.name == "QueryHandler":
                        self.query_handler_count += 1
                        logger.info("Incremented query_handler_count to %d", self.query_handler_count)
                    logger.info("Selected speaker: %s", agent.name)
                    return agent

            logger.warning("No valid speaker selected, defaulting to QueryHandler")
            self.query_handler_count += 1
            return self.agents[0]
        except Exception as e:
            logger.error("Speaker selection failed: %s", e)
            return self.agents[0]

    async def a_run_chat(self, message, sender=None):
        """Custom chat loop to manage agent interactions."""
        try:
            logger.info("Starting a_run_chat with message: %s, sender=%s", message, sender)
            self.query_handler_count = 0  # Reset counter
            self.append(message)
            round_count = 0

            while round_count < self.max_round:
                last_speaker = self.agents[0] if round_count == 0 else None
                speaker = await self.a_select_speaker(last_speaker=last_speaker, selector=self)
                
                if speaker is None:
                    logger.info("No speaker selected, terminating chat")
                    break

                logger.info("Round %d: Selected speaker: %s", round_count + 1, speaker.name)
                success, reply = await speaker.a_generate_reply(messages=self.messages, sender=self)
                logger.info("Reply from %s: %s", speaker.name, reply)

                if not success:
                    logger.error("Failed to generate reply from %s", speaker.name)
                    return TextMessage(content="ERROR: Failed to generate reply", source="System")

                self.append(reply)
                
                if isinstance(reply, TextMessage) and ("INVALID" in reply.content or "ERROR" in reply.content):
                    logger.info("Invalid query or error detected, stopping chat")
                    return reply

                if speaker.name == "RAGRetriever":
                    logger.info("Answer provided by RAGRetriever, stopping chat")
                    return reply

                round_count += 1

            logger.warning("Max rounds reached: %d", self.max_round)
            return TextMessage(content="ERROR: Max rounds reached without answer", source="System")
        except Exception as e:
            logger.error("Chat loop failed: %s", e)
            return TextMessage(content=f"ERROR: Chat failed - {e}", source="System")

    async def reset(self):
        """Reset the chat history."""
        self.messages = []
        self.query_handler_count = 0
        logger.info("Reset CustomSelectorGroupChat messages and query_handler_count")

# === Main Async Chat Orchestrator ===
async def main():
    # Load and process PDF
    faq_path = r'C:\Users\sahee\OneDrive\Desktop\Lealabs\WEEK 3\May 22\FAQs.pdf'
    try:
        texts = load_faq_text(faq_path)
        chunks = split_chunks(texts)
        collection = setup_chromadb(chunks)
    except Exception as e:
        print("❌ Failed to process PDF: %s" % e)
        logger.error("PDF processing failed: %s", e)
        return

    # Create Gemini-compatible model client
    model_client = GeminiModelClient(api_key=GOOGLE_API_KEY)

    # Agents
    query_handler = QueryHandlerAgent(
        name="QueryHandler",
        model_client=model_client,
        llm_config=False,
        system_message="You validate and rephrase user queries for clarity."
    )

    retriever_agent = RAGRetrieverAgent(
        name="RAGRetriever",
        collection=collection,
        model_client=model_client,
        llm_config=False,
        system_message="You retrieve and answer questions using a RAG system."
    )

    # Group Chat
    try:
        groupchat = CustomSelectorGroupChat(
            agents=[query_handler, retriever_agent],
            model_client=model_client,
            selector_prompt="Select the next agent to respond based on the last message. QueryHandler validates/rephrases queries, RAGRetriever answers them. User can terminate with 'exit'.",
            termination_condition=lambda msg: "exit" in msg.get("content", "").lower(),
            allow_repeated_speaker=True
        )
        logger.info("Initialized CustomSelectorGroupChat")
        await groupchat.reset()
    except Exception as e:
        logger.error("Failed to initialize CustomSelectorGroupChat: %s", e)
        print("❌ Failed to initialize group chat: %s" % e)
        return

    # Interactive chat
    print("🚀 Starting FAQ BOT. Type your question (or 'exit' to quit):")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            print("✅ Exiting FAQ BOT.")
            logger.info("FAQ BOT closed.")
            break

        try:
            message = TextMessage(content=question, source="User")
            logger.info("Created TextMessage: %s", message)
            reply = await groupchat.a_run_chat(message=message, sender=query_handler)
            logger.info("Chat reply: %s", reply)
            print("Answer: %s" % reply.content)
        except Exception as e:
            logger.error("Chat failed: %s", e)
            print("❌ Failed to get answer: %s" % e)

# === Entry Point ===
if __name__ == "__main__":
    asyncio.run(main())