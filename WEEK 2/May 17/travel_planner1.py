import autogen

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

# Define agents with system messages and LLM config
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
    system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
)

travel_summary_agent = autogen.AssistantAgent(
    name="travel_summary_agent",
    llm_config=llm_config,
    system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
)

# Termination condition and group chat config
groupchat = autogen.GroupChat(
    agents=[planner_agent, local_agent, language_agent, travel_summary_agent],
    messages=[],
    max_round=10,
    speaker_selection_method="round_robin",
)

# Human proxy to initiate conversation
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

# Start the group chat task
user_proxy.initiate_chat(
    group_manager,
    message="Plan a 3 day trip to Nepal."
)
