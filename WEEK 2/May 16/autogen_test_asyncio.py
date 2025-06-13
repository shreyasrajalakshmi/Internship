import autogen
import asyncio

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

async def run_task(user_proxy, assistant, task):
    print(f"Running task: {task.strip()[:30]}...")
    await user_proxy.a_initiate_chat(assistant, message=task)
    print("Task completed.\n")

async def main():
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
        code_execution_config={"work_dir": "new1", "use_docker": False},
        llm_config=llm_config,
        system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
    )

    task1 = """
    Write a Python script that:
    - Checks if the string 'madam' is a palindrome
    - Prints 'madam is a palindrome' if it is, or 'madam is not a palindrome' otherwise
    - DO NOT include any user input() — the string must be hardcoded
    - Save the script as 'palindrome.py' in the working directory mentioned.
    """

    task2 = """
    Write a Python script to print even numbers starting from 1 to 50 and save it as 'even_50.py' in the working directory mentioned.
    """

    # Run tasks sequentially, waiting for one to finish before starting the next
    #await run_task(user_proxy, assistant, task1)
    await run_task(user_proxy, assistant, task2)

if __name__ == "__main__":
    asyncio.run(main())
