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
        self.max_attempts = 3
        self.attempts = 0

    def _clean_code(self, raw_code: str) -> str:
        """Remove Markdown code block syntax if present."""
        # Remove leading/trailing whitespace
        code = raw_code.strip()
        
        # Remove Python markdown blocks
        if code.startswith("```python") and code.endswith("```"):
            return "\n".join(code.split("\n")[1:-1])
        elif code.startswith("```") and code.endswith("```"):
            return "\n".join(code.split("\n")[1:-1])
        
        # Remove any remaining backticks that might be on their own line
        code = "\n".join([line for line in code.split("\n") if not line.strip() == "```"])
        
        return code

    async def a_generate_reply(self, sender):
        last_msg = self.chat_messages[sender][-1]["content"]
        
        # Respond to debugger's approval request
        if "DEBUGGER: Code looks good!" in last_msg:
            return {
                "name": self.name,
                "content": "CODER: Okay, terminating the program. Have a great day!",
                "is_termination_msg": True
            }
            
        # Respond to fix requests
        if "DEBUGGER: Found issues:" in last_msg:
            self.attempts += 1
            if self.attempts >= self.max_attempts:
                return {
                    "name": self.name,
                    "content": "CODER: Max fixes attempted. Terminating.",
                    "is_termination_msg": True
                }
            
            original_task = self.chat_messages[sender][0]["content"]
            prompt = f"Fix this code based on feedback:\n{last_msg}\nOriginal task:\n{original_task}"
            
            print(f"[{self.name}] Generating fixed code...")
            raw_code = generate_with_retry(prompt)
            clean_code = self._clean_code(raw_code)
            result = self.executor.run(clean_code)
            
            return {
                "name": self.name,
                "content": f"CODE_SUBMISSION:\n{clean_code}\nEXECUTION_RESULT:\n{result}",
                "is_termination_msg": False
            }
        
        # Initial code generation
        if not any(msg["name"] == self.name for msg in self.chat_messages[sender]):
            print(f"[{self.name}] Generating initial code...")
            raw_code = generate_with_retry(last_msg)
            clean_code = self._clean_code(raw_code)
            result = self.executor.run(clean_code)
            
            return {
                "name": self.name,
                "content": f"CODE_SUBMISSION:\n{clean_code}\nEXECUTION_RESULT:\n{result}",
                "is_termination_msg": False
            }
            
        return None

class DebuggerAgent(AssistantAgent):
    def __init__(self, name, linter: PylintLinter):
        super().__init__(name=name)
        self.linter = linter
        self.approved = False

    async def a_generate_reply(self, sender):
        if self.approved:
            return None
            
        last_msg = self.chat_messages[sender][-1]["content"]
        
        if "CODE_SUBMISSION:" in last_msg:
            code = last_msg.split("CODE_SUBMISSION:")[1].split("EXECUTION_RESULT:")[0].strip()
            
            print(f"[{self.name}] Linting and debugging code...")
            lint_result = self.linter.lint(code)

            if not lint_result or ("no issues found" in lint_result.lower()):
                self.approved = True
                return {
                    "name": self.name,
                    "content": "DEBUGGER: Code looks good! Shall we terminate the program?",
                    "is_termination_msg": False
                }
            
            return {
                "name": self.name,
                "content": f"DEBUGGER: Found issues:\n{lint_result}\nPlease fix them.",
                "is_termination_msg": False
            }
        
        # Acknowledge termination confirmation
        if "CODER: Okay, terminating" in last_msg:
            self.approved = True
            return {
                "name": self.name,
                "content": "DEBUGGER: All done! Have a good day!",
                "is_termination_msg": True
            }
            
        return None
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
        max_round=6,  # Reduced from 100 to prevent endless loops
        speaker_selection_method="round_robin"
    )
    
    manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=False,
    is_termination_msg=lambda msg: (
        "Have a great day!" in msg.get("content", "") or
        "All done!" in msg.get("content", "") or
        "Terminating." in msg.get("content", "")
    )
)
    
    await user.a_initiate_chat(manager, message=task)
    print("Conversation completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())