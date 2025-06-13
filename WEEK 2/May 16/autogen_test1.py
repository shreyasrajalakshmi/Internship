import autogen

config_list = [
    {
        'model': 'gemini-2.0-flash',
        'api_key': 'AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg',  # Replace with your actual API key
        "api_type": "google",
    }
]

llm_config = {
    "seed": 42,
    "config_list": config_list,
    "temperature": 0
}

assistant = autogen.AssistantAgent(
    name="CTO",
    llm_config=llm_config,
    system_message="Chief technical officer of a tech company"
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "new", "use_docker": False},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

task = """
Write python code that prints numbers 1 to 50, and then save this code into a file named 'output_50.py' in the working directory.
"""

# Start the conversation by sending the task message from user_proxy to assistant
last_message = user_proxy.initiate_chat(assistant, message=task)

# Now loop until termination is detected or max cycles reached (for safety)
max_turns = 20
turn = 0

while turn < max_turns:
    turn += 1
    # user_proxy sends message to assistant and gets reply
    reply_from_assistant = assistant.receive(last_message, user_proxy)

    # assistant replies back, now user_proxy receives
    reply_from_user_proxy = user_proxy.receive(reply_from_assistant, assistant)

    # Check if termination message detected from user_proxy reply
    if user_proxy.is_termination_msg(reply_from_user_proxy):
        print("\nTermination detected. Stopping conversation.")
        break

    # Prepare for next loop cycle
    last_message = reply_from_user_proxy

else:
    print("\nReached max turns without termination.")

