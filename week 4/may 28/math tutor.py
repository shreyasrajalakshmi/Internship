import asyncio
import sympy as sp
import google.generativeai as genai
import ast

# Your Gemini API key here
GOOGLE_API_KEY = "AIzaSyCHy80eWH_N7Q9Xc0niq9OpxdNKaCoJmBQ"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

x = sp.Symbol('x')

def solve_math_problem(problem: str) -> str:
    try:
        if '=' in problem:
            lhs, rhs = problem.split('=')
            equation = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
            solution = sp.solve(equation, x)
            return f"Solution: {solution}"
        elif 'd/dx' in problem or 'differentiate' in problem.lower():
            expr = problem.lower().replace('d/dx', '').replace('differentiate', '').strip()
            derivative = sp.diff(sp.sympify(expr), x)
            return f"Derivative: {derivative}"
        else:
            result = sp.sympify(problem)
            return f"Result: {result}"
    except Exception as e:
        return f"Error solving problem: {e}"

def verify_solution(problem: str, solution: str) -> str:
    try:
        if '=' in problem:
            lhs, rhs = problem.split('=')
            lhs_expr = sp.sympify(lhs)
            rhs_expr = sp.sympify(rhs)
            sols_str = solution.replace("Solution:", "").strip()
            sols = ast.literal_eval(sols_str)
            is_correct = all(sp.simplify(lhs_expr.subs(x, s) - rhs_expr.subs(x, s)) == 0 for s in sols)
            return "✅ Verified correct!" if is_correct else "❌ Verification failed."
        else:
            return "Verification only supports equations."
    except Exception as e:
        return f"Error verifying solution: {e}"

async def query_gemini(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()

async def main():
    print("Smart Math Tutor (Google Gemini + SymPy)")
    print("Type 'exit' to quit.")
    print("You can enter equations (e.g., x^2 - 4 = 0), derivatives (e.g., differentiate x**2), or arithmetic (e.g., 2 + 2*5).")

    while True:
        problem = input("\nEnter math problem: ")
        if problem.lower() == 'exit':
            print("Goodbye!")
            break

        solution = solve_math_problem(problem)
        print("🧮 Problem Solver:", solution)

        verification = verify_solution(problem, solution)
        print("✅ Verifier:", verification)

        prompt = f"Explain this math problem and solution:\nProblem: {problem}\nSolution: {solution}"
        explanation = await query_gemini(prompt)
        print("💡 Gemini Explanation:", explanation)

if __name__ == "__main__":
    asyncio.run(main())
