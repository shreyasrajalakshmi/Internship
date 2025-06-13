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
    code_execution_config={"work_dir": "web", "use_docker": False},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

# task = """
# Write python code that prints numbers 1 to 50, and then save this code into a file named 'output_50_1.py' in the working directory.
# """
task = """
Write a Python script that checks if the string 'madam' is a palindrome and save it as 'palindrome.py' in the working directory.
"""
user_proxy.initiate_chat(
    assistant,
    message=task
)

# task2 = """
# Change the code in the file named 'output_50.py' you just created to instead output numbers 1 to 100 and save it as 'output_100.py'.
# """

# user_proxy.initiate_chat(
#     assistant,
#     message=task2
# )