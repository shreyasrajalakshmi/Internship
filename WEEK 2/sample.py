from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

custom_model_info = {
    "name": "llama3-70b-8192",
    "family": "llama3",
    "max_tokens": 8192,
    "structured_output": False,
    "requires_system_message": True,
    "tool_choice_support": True,
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "vision": False,
    "function_calling": True,  # <-- set True to avoid the error
    "json_output": False,
}



model_client = OpenAIChatCompletionClient(
    model="llama3-70b-8192",  # Groq supports LLaMA3 models
    api_key="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",
    base_url="https://api.groq.com/openai/v1", 
     model_info=custom_model_info, # Groq's OpenAI-compatible endpoint
)

async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,
)

async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York?"))
    await model_client.close()

asyncio.run(main())
