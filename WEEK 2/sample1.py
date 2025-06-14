from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

# Minimal model info for llama3-70b without function calling
custom_model_info = {
    "name": "llama3-70b-8192",
    "family": "llama3",
    "max_tokens": 8192,
    "structured_output": False,
    "requires_system_message": True,
    "tool_choice_support": False,
    "parallel_tool_calls": False,
    "tool_choice": "none",
    "vision": False,
    "function_calling": True,
    "json_output": False,
}

model_client = OpenAIChatCompletionClient(
    model="llama3-70b-8192",
    api_key="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",  # Your Groq API key
    base_url="https://api.groq.com/openai/v1",
    model_info=custom_model_info,
)

# Create a basic AssistantAgent without tools or function calls
agent = AssistantAgent(
    name="simple_agent",
    model_client=model_client,
    tools=[],  # no tools
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=False,
    model_client_stream=True,
)

async def main():
    # Run a simple chat task
    await Console(agent.run_stream(task="What is the benefit of Gen AI RAG?"))
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
