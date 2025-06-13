import asyncio
import os
import tempfile
import subprocess
import time
from typing import Any, Dict
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from autogen import UserProxyAgent, AssistantAgent
from autogen.agentchat import GroupChat, GroupChatManager

# === Configure Gemini LLM ===
GOOGLE_API_KEY = "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg"  # Consider using environment variables for API keys
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# === TOOLS ===
class PythonExecutor:
    def run(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
            f.write(code)
            temp_name = f.name
        try:
            result = subprocess.run(["python", temp_name], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            output = result.stdout + result.stderr
        except Exception as e:
            output = str(e)
        finally:
            os.unlink(temp_name)
        return output.strip()

class PylintLinter:
    def lint(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
            f.write(code)
            temp_name = f.name
        try:
            # Updated pylint command to use specific checks instead of 'errors,warnings'
            result = subprocess.run(["pylint", temp_name, "--disable=all", "--enable=E,W"],
                                  capture_output=True, 
                                  text=True)
            output = result.stdout + result.stderr
        except Exception as e:
            output = str(e)
        finally:
            os.unlink(temp_name)
        return output.strip()

# === GEMINI CALL WRAPPER WITH RETRY ===
def generate_with_retry(prompt: str, retries: int = 5, delay: int = 60) -> str:
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except ResourceExhausted as e:
            print(f"[Gemini] Quota exceeded. Retrying in {delay} seconds... ({attempt+1}/{retries})")
            time.sleep(delay)
        except Exception as e:
            print(f"[Gemini] Unexpected error: {e}")
            break
    return "ERROR: Failed to get a response from Gemini after multiple retries."

# === AGENTS ===
class CoderAgent(AssistantAgent):
    def __init__(self, name, executor: PythonExecutor):
        super().__init__(name=name)
        self.executor = executor
        self.task_completed = False

    def _clean_code(self, raw_code: str) -> str:
        """Remove Markdown code block syntax if present."""
        if raw_code.startswith("```python"):
            return "\n".join(raw_code.split("\n")[1:-1])
        elif raw_code.startswith("```"):
            return "\n".join(raw_code.split("\n")[1:-1])
        return raw_code

    async def a_generate_reply(self, sender):
        if self.task_completed:
            return None  # Don't respond after task is complete
            
        last_msg = self.chat_messages[sender][-1]["content"]
        
        # Don't respond to debugger's success message
        if "Code looks good!" in last_msg:
            self.task_completed = True
            return None
            
        prompt = f"Write Python code to fulfill the task:\n\n{last_msg}"
        print(f"[{self.name}] Generating code...")
        raw_code = generate_with_retry(prompt)
        
        # Check if Gemini is asking for clarification instead of providing code
        if not any(line.strip().startswith(('def ', 'class ', 'import ', 'print(')) for line in raw_code.split('\n')):
            return {
                "name": self.name,
                "content": "I need a clear task description to write code. Please specify what you want me to do.",
                "is_termination_msg": True
            }
            
        clean_code = self._clean_code(raw_code)
        result = self.executor.run(clean_code)
        return {
            "name": self.name,
            "content": f"Code:\n{clean_code}\n\nExecution Result:\n{result}",
            "is_termination_msg": False
        }

class DebuggerAgent(AssistantAgent):
    def __init__(self, name, linter: PylintLinter):
        super().__init__(name=name)
        self.linter = linter

    async def a_generate_reply(self, sender):
        last_msg = self.chat_messages[sender][-1]["content"]
        
        # Check if this is a successful execution result
        if "Execution Result:" in last_msg and "Code:" in last_msg:
            code_start = last_msg.find("Code:\n") + len("Code:\n")
            code_end = last_msg.find("\n\nExecution Result:")
            code = last_msg[code_start:code_end].strip()

            print(f"[{self.name}] Linting and debugging code...")
            lint_result = self.linter.lint(code)

            if "error" in lint_result.lower() or "warning" in lint_result.lower():
                return {
                    "name": self.name,
                    "content": f"Found some issues via linting:\n{lint_result}\n\nPlease fix them.",
                    "is_termination_msg": False
                }
            else:
                return {
                    "name": self.name,
                    "content": "Code looks good! Task completed successfully.",
                    "is_termination_msg": True  # This will end the conversation
                }
        
        return None  # Don't respond to other types of messages
# === MAIN ===
async def main():
    task = input("Enter the task you want the agents to work on: ").strip()
    if not task:
        print("No task provided. Exiting.")
        return

    executor = PythonExecutor()
    linter = PylintLinter()

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
        max_consecutive_auto_reply=3  # Limit automatic replies
    )
    
    coder = CoderAgent(name="Coder", executor=executor)
    debugger = DebuggerAgent(name="Debugger", linter=linter)

    groupchat = GroupChat(
        agents=[user, coder, debugger],
        messages=[],
        max_round=50,  # Reduced from 100 to prevent endless loops
        speaker_selection_method="round_robin"
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=False,
        is_termination_msg=lambda msg: "is_termination_msg" in msg and msg["is_termination_msg"]
    )
    
    await user.a_initiate_chat(manager, message=task)
    print("Conversation completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())