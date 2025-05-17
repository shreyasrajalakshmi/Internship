import autogen

# ----------------Step 1: Define the shared LLM configuration for all agents ----------------
config_list = [
    {
        "model": "gemini-1.5-flash",
        "api_key": "AIzaSyCHy80eWH_N7Q9Xc0niq9OpxdNKaCoJmBQ",  
        "api_type": "google",
    }
]

llm_config = {
    "seed": 42,          
    "temperature": 0,     
    "config_list": config_list,
}

# ----------------Step 2: Create agents with specific roles and system messages----------------
planner_agent = autogen.AssistantAgent(
    name="planner_agent",
    llm_config=llm_config,
    system_message=(
        "You are a helpful assistant that can suggest a travel plan for a user "
        "based on their request."
    ),
)

local_agent = autogen.AssistantAgent(
    name="local_agent",
    llm_config=llm_config,
    system_message=(
        "You are a helpful assistant that can suggest authentic and interesting "
        "local activities or places to visit for a user and can utilize any context information provided."
    ),
)

language_agent = autogen.AssistantAgent(
    name="language_agent",
    llm_config=llm_config,
    system_message=(
        "You are a helpful assistant that can review travel plans, providing feedback "
        "on important/critical tips about how best to address language or communication challenges "
        "for the given destination. If the plan already includes language tips, mention that it is satisfactory, with rationale."
    ),
)

travel_summary_agent = autogen.AssistantAgent(
    name="travel_summary_agent",
    llm_config=llm_config,
    system_message=(
        "You are a helpful assistant that can take all suggestions and advice from other agents "
        "and provide a detailed final travel plan. Ensure the final plan is integrated and complete. "
        "YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete, respond with TERMINATE."
    ),
)

# ----------------Step 3: Setup the group chat with agents and conversation rules----------------
groupchat = autogen.GroupChat(
    agents=[planner_agent, local_agent, language_agent, travel_summary_agent],
    messages=[],
    max_round=10,                    # Max turns in the chat before stopping
    speaker_selection_method="round_robin",  # Agents speak in fixed order
)

# ----------------Step 4: Initialize the manager that runs the group chat----------------
group_manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# ----------------Step 5: Create a User Proxy Agent that initiates conversation and monitors termination----------------
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Fully automated; no human input after start
    code_execution_config=False,
    is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
)

# ----------------Step 6: Start the conversation with user's travel request----------------
user_proxy.initiate_chat(
    group_manager,
    message="Plan a 3 day trip to Nepal."
)
