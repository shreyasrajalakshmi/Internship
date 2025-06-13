import asyncio
import os
import tempfile
import subprocess
from typing import Any, Dict
import google.generativeai as genai
from autogen import UserProxyAgent, AssistantAgent
from autogen.agentchat import GroupChat, GroupChatManager

# === Configure Gemini LLM ===
GOOGLE_API_KEY = "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# === TOOLS ===
class PythonExecutor:
    def run(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
            f.write(code)
            temp_name = f.name
        # file closed here

        try:
            result = subprocess.run(["python", temp_name], capture_output=True, text=True, timeout=10)
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
        # file closed here

        try:
            result = subprocess.run(["pylint", temp_name, "--disable=all", "--enable=errors,warnings"], capture_output=True, text=True)
            output = result.stdout + result.stderr
        except Exception as e:
            output = str(e)
        finally:
            os.unlink(temp_name)
        return output.strip()


# === AGENTS ===
class CoderAgent(AssistantAgent):
    def __init__(self, name, executor: PythonExecutor):
        super().__init__(name=name)
        self.executor = executor

    async def a_generate_reply(self, sender):
        last_msg = self.chat_messages[sender][-1]["content"]
        prompt = f"Write Python code to fulfill the task:\n\n{last_msg}"
        print(f"[{self.name}] Generating code...")
        response = model.generate_content(prompt)
        code = response.text
        result = self.executor.run(code)
        return {
            "name": self.name,
            "content": f"Code:\n{code}\n\nExecution Result:\n{result}",
            "is_termination_msg": False
        }

class DebuggerAgent(AssistantAgent):
    def __init__(self, name, linter: PylintLinter):
        super().__init__(name=name)
        self.linter = linter
        self.acknowledged = False

    async def a_generate_reply(self, sender):
        if self.acknowledged:
            return {
                "name": self.name,
                "content": "Thanks! Code looks good now. Ending chat.",
                "is_termination_msg": True
            }

        last_msg = self.chat_messages[sender][-1]["content"]
        code_start = last_msg.find("Code:\n") + len("Code:\n")
        code_end = last_msg.find("\n\nExecution Result:")
        code = last_msg[code_start:code_end].strip()

        print(f"[{self.name}] Linting and debugging code...")
        lint_result = self.linter.lint(code)

        if "error" in lint_result.lower() or "warning" in lint_result.lower():
            reply = f"Found some issues via linting:\n{lint_result}\n\nPlease fix them."
        else:
            self.acknowledged = True
            reply = "No major issues found. Code seems good. Thanks!"

        return {
            "name": self.name,
            "content": reply,
            "is_termination_msg": self.acknowledged
        }

# === MAIN ===
async def main():
    task = input("Enter the task you want the agents to work on: ")

    executor = PythonExecutor()
    linter = PylintLinter()

    user = UserProxyAgent(name="User", human_input_mode="NEVER", code_execution_config={"use_docker": False})

    coder = CoderAgent(name="Coder", executor=executor)
    debugger = DebuggerAgent(name="Debugger", linter=linter)

    groupchat = GroupChat(
        agents=[user, coder, debugger],
        messages=[],
        max_round=100,
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=False)

    await user.a_initiate_chat(manager, message=task)

if __name__ == "__main__":
    asyncio.run(main())
