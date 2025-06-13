import os
import sympy as sp
from autogen import AssistantAgent, UserProxyAgent
import google.generativeai as genai
from autogen.agentchat import GroupChat, GroupChatManager
import json
import asyncio
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('math_tutor.log')]
)
logger = logging.getLogger(__name__)

# Gemini API Setup
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")  # Replace with your Google API key or set GEMINI_API_KEY environment variable
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error("Failed to configure Gemini API: %s", e)
    raise

# Gemini Retry Mechanism
async def gemini_generate(prompt: str, model_name="gemini-1.5-flash", retries: int = 5, delay: int = 60) -> str:
    """Generate content using Gemini API with retry logic."""
    model = genai.GenerativeModel(model_name)
    for i in range(retries):
        try:
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(prompt))
            logger.info("Gemini API call successful on attempt %d", i + 1)
            return response.text
        except genai.types.generation_types.BlockedPromptException as e:
            logger.error("Gemini API blocked prompt: %s", e)
            return "ERROR: Prompt blocked by Gemini API."
        except genai.types.generation_types.StopCandidateException as e:
            logger.error("Gemini API stopped candidate: %s", e)
            return "ERROR: Response generation stopped by Gemini API."
        except Exception as e:
            if "429" in str(e):
                logger.error("Gemini API quota exceeded: %s", e)
                return "ERROR: Gemini API quota exceeded."
            logger.warning("Gemini error. Retrying in %ds (%d/%d): %s", delay, i + 1, retries, e)
            await asyncio.sleep(delay)
    logger.error("Gemini API failed after %d retries", retries)
    return "ERROR: Gemini API failed after retries."

# SymPy Tool for solving math problems
def solve_math_problem(problem):
    try:
        x = sp.Symbol('x')
        if "solve" in problem.lower():
            eq_str = problem.lower().replace("solve", "").strip()
            eq = sp.sympify(eq_str)
            solutions = sp.solve(eq, x)
            latex_solutions = [sp.latex(sol) for sol in solutions]
            return {
                "solution": f"Solutions: {solutions}",
                "latex": f"Solutions: \\({', '.join(latex_solutions)}\\)",
                "raw_solutions": solutions
            }
        elif "differentiate" in problem.lower():
            expr_str = problem.lower().replace("differentiate", "").strip()
            expr = sp.sympify(expr_str)
            derivative = sp.diff(expr, x)
            return {
                "solution": f"Derivative: {derivative}",
                "latex": f"Derivative: \\({sp.latex(derivative)}\\)",
                "raw_solution": derivative
            }
        elif "+" in problem or "-" in problem or "*" in problem or "/" in problem:
            result = sp.sympify(problem).evalf()
            return {
                "solution": f"Result: {result}",
                "latex": f"Result: \\({sp.latex(result)}\\)",
                "raw_solution": result
            }
        else:
            return {"error": "Unsupported problem type"}
    except Exception as e:
        logger.error("Error solving problem: %s", e)
        return {"error": f"Error solving problem: {str(e)}"}

# Calculator Tool for verifying solutions
def verify_solution(problem, solution_data):
    try:
        if "error" in solution_data:
            return f"Cannot verify: {solution_data['error']}"
        x = sp.Symbol('x')
        if "solve" in problem.lower():
            eq_str = problem.lower().replace("solve", "").strip()
            eq = sp.sympify(eq_str)
            solutions = solution_data["raw_solutions"]
            for sol in solutions:
                result = eq.subs(x, sol)
                if not sp.simplify(result) == 0:
                    return f"Verification failed: Substituting x={sol} does not satisfy {eq_str}"
            return "Solution verified correct. TERMINATE"
        elif "differentiate" in problem.lower():
            expr_str = problem.lower().replace("differentiate", "").strip()
            expr = sp.sympify(expr_str)
            expected_derivative = sp.diff(expr, x)
            if sp.simplify(expected_derivative - solution_data["raw_solution"]) == 0:
                return "Derivative verified correct. TERMINATE"
            return f"Verification failed: Expected {expected_derivative}, got {solution_data['raw_solution']}"
        elif "Result" in solution_data["solution"]:
            expected = sp.sympify(problem).evalf()
            if sp.simplify(expected - solution_data["raw_solution"]) == 0:
                return "Arithmetic result verified correct. TERMINATE"
            return f"Verification failed: Expected {expected}, got {solution_data['raw_solution']}"
        return "Unsupported verification type"
    except Exception as e:
        logger.error("Error verifying solution: %s", e)
        return f"Error verifying solution: {str(e)}"

# Agent configurations
function_config = {
    "config_list": [
        {
            "model": "gemini-1.5-flash",
            "api_type": "google",
            "api_key": GOOGLE_API_KEY
        }
    ],
    "functions": [
        {
            "name": "solve_math_problem",
            "description": "Solves a math problem using SymPy",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string", "description": "The math problem to solve"}
                },
                "required": ["problem"]
            }
        },
        {
            "name": "verify_solution",
            "description": "Verifies a math solution using SymPy",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string", "description": "The original math problem"},
                    "solution_data": {"type": "object", "description": "The solution data to verify"}
                },
                "required": ["problem", "solution_data"]
            }
        }
    ]
}

# Manager config without functions
manager_config = {
    "config_list": [
        {
            "model": "gemini-1.5-flash",
            "api_type": "google",
            "api_key": GOOGLE_API_KEY
        }
    ]
}

# Custom GroupChatManager with termination on "TERMINATE"
class CustomGroupChatManager(GroupChatManager):
    async def a_run_chat(self, messages=None, sender=None, config=None):
        """Override to stop chat when 'TERMINATE' is in a message."""
        groupchat = self._groupchat
        messages = messages or []
        if not isinstance(messages, list):
            messages = [messages]
        groupchat.messages.extend(messages)
        
        round_count = 0
        while round_count < groupchat.max_round:
            speaker = groupchat.select_speaker(
                last_speaker=sender,
                selector=self
            )
            if not speaker:
                logger.info("No speaker selected, terminating chat")
                break

            logger.info("Round %d: Selected speaker: %s", round_count + 1, speaker.name)
            reply = await speaker.a_generate_reply(
                messages=groupchat.messages,
                sender=self
            )

            if reply is None:
                logger.info("No reply generated, terminating chat")
                break

            groupchat.append(reply, speaker)

            # Check for termination condition
            if isinstance(reply, dict) and "content" in reply:
                if "TERMINATE" in reply["content"].upper():
                    logger.info("TERMINATE detected, stopping chat")
                    break

            round_count += 1

        logger.info("Chat completed after %d rounds", round_count)
        return groupchat.messages[-1] if groupchat.messages else None

# Initialize agents
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={
        "work_dir": "math_tutor",
        "use_docker": False
    },
)

problem_solver = AssistantAgent(
    name="ProblemSolver",
    llm_config=function_config,
    system_message="You are a math expert. Use the solve_math_problem function to solve math problems and provide solutions in both plain text and LaTeX format. If the Verifier indicates an error, revise your solution.",
    function_map={"solve_math_problem": solve_math_problem}
)

verifier = AssistantAgent(
    name="Verifier",
    llm_config=function_config,
    system_message="You are a math verifier. Use the verify_solution function to check the correctness of solutions. If correct, output 'Solution verified correct. TERMINATE'. If incorrect, explain the issue clearly to guide the ProblemSolver.",
    function_map={"verify_solution": verify_solution}
)

# Group chat configuration
group_chat = GroupChat(
    agents=[problem_solver, verifier],
    messages=[],
    max_round=10,
    speaker_selection_method="round_robin"
)

manager = CustomGroupChatManager(
    groupchat=group_chat,
    llm_config=manager_config
)

# Async function to run the math tutor
async def run_math_tutor(problem):
    try:
        logger.info("Running math tutor for problem: %s", problem)
        await user_proxy.a_initiate_chat(
            recipient=manager,
            message=f"Solve this math problem: {problem}"
        )
        logger.info("Completed math tutor run for problem: %s", problem)
    except Exception as e:
        logger.error("Math tutor run failed: %s", e)
        print(f"❌ Failed to solve problem: {e}")

# Main async orchestrator
async def main():
    print("🚀 Smart Math Tutor: Enter a math problem (e.g., 'Solve x^2 - 4 = 0', 'Differentiate x^2', '2 + 2').")
    print("Type 'terminate' or 'exit' to quit.")
    
    while True:
        problem = input("\nEnter math problem: ").strip()
        logger.info("User input: %s", problem)
        
        if problem.lower() in ["terminate", "exit"]:
            print("✅ Exiting Smart Math Tutor.")
            logger.info("Smart Math Tutor terminated by user.")
            break
        
        if not problem:
            print("⚠️ Please enter a valid math problem or 'terminate' to quit.")
            logger.warning("Empty input received")
            continue
        
        print(f"\nSolving: {problem}")
        await run_math_tutor(problem)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())