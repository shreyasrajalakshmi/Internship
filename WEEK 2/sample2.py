import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

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
    api_key="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",
    base_url="https://api.groq.com/openai/v1",
    model_info=custom_model_info,
)

agent = AssistantAgent(
    name="chat_agent",
    model_client=model_client,
    tools=[],
    system_message="You are a helpful assistant. Chat naturally.",
    reflect_on_tool_use=False,
    model_client_stream=False,
)

async def main():
    print("Start chatting with the assistant. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = await agent.run(task=user_input)
        print("\nAssistant:", result.messages[-1].content, "\n")

    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
