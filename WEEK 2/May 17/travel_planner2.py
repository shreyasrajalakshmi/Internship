import autogen
import asyncio

# Common Gemini configuration using high-level API
config_list = [
    {
        "model": "gemini-1.5-flash",
        "api_key": "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg",  # Replace with your actual API key
        "api_type": "google",
    }
]

llm_config = {
    "seed": 42,
    "temperature": 0,
    "config_list": config_list,
}

async def main():
    # Define agents
    planner_agent = autogen.AssistantAgent(
        name="planner_agent",
        llm_config=llm_config,
        system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
    )

    local_agent = autogen.AssistantAgent(
        name="local_agent",
        llm_config=llm_config,
        system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
    )

    language_agent = autogen.AssistantAgent(
        name="language_agent",
        llm_config=llm_config,
        system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination.",
    )

    travel_summary_agent = autogen.AssistantAgent(
        name="travel_summary_agent",
        llm_config=llm_config,
        system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When complete, respond with TERMINATE.",
    )

    # Group chat setup
    groupchat = autogen.GroupChat(
        agents=[planner_agent, local_agent, language_agent, travel_summary_agent],
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin",
    )

    group_manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        llm_config=llm_config,
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
    )

    # 🧠 Use the async version of the method
    await user_proxy.a_initiate_chat(
        group_manager,
        message="Plan a 3 day trip to Nepal."
    )

# 🔁 This line runs the async function
if __name__ == "__main__":
    asyncio.run(main())
