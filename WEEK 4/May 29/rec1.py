import os
import json
import asyncio
import random
from autogen import AssistantAgent, UserProxyAgent
import google.generativeai as genai
from autogen.agentchat import GroupChat, GroupChatManager

# Gemini API Setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")  # Replace or set GOOGLE_API_KEY
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise Exception(f"Failed to configure Gemini: {str(e)}")

# Gemini Retry Mechanism
async def gemini_generate(prompt: str, model_name="gemini-1.5-flash", retries: int = 5, delay: int = 1) -> str:
    model = genai.GenerativeModel(model_name)
    for _ in range(retries):
        try:
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(prompt))
            return response.text
        except Exception as e:
            if "429" in str(e):
                return "ERROR: Gemini API quota exceeded."
            await asyncio.sleep(delay)
    return "ERROR: Gemini API failed after retries."

# Load ingredients
def load_ingredients():
    try:
        with open("ingredients.json", "r") as f:
            return json.load(f)
    except Exception:
        return []

# Generate recipe
async def generate_recipe(user_input: str, feedback: str = None):
    try:
        ingredients_db = load_ingredients()
        if not ingredients_db:
            return {"error": "Ingredients database empty or not found"}

        # Parse input
        input_lower = user_input.lower()
        dietary = "vegan" if "vegan" in input_lower else ""
        servings = 2
        if "for" in input_lower:
            try:
                servings = int(input_lower.split(" ")[-1])
            except (IndexError, ValueError):
                pass
        if "snack" in input_lower:
            servings = 1

        # Filter ingredients
        valid_ingredients = [
            ing for ing in ingredients_db if not dietary or dietary in [d.lower() for d in ing["dietary"]]
        ]
        if not valid_ingredients:
            return {"error": "No ingredients match dietary preference."}

        # Select ingredients
        selected = random.sample(valid_ingredients, min(3, len(valid_ingredients)))
        recipe_ingredients = [
            {"name": ing["name"], "quantity_g": random.randint(50, 150) if "snack" in input_lower else random.randint(100, 250)}
            for ing in selected
        ]

        # Adjust based on feedback
        if feedback:
            if "calorie" in feedback.lower():
                recipe_ingredients = [
                    {"name": ing["name"], "quantity_g": int(ing["quantity_g"] * 0.8)} for ing in recipe_ingredients
                ]
            if "protein" in feedback.lower():
                for ing in recipe_ingredients:
                    if ing["name"] in ["tofu", "soy sauce"]:
                        ing["quantity_g"] = int(ing["quantity_g"] * 1.3)

        # Generate instructions
        dietary_name = dietary.capitalize() or "Standard"
        meal_type = "Snack" if "snack" in input_lower else "Stir-Fry"
        prompt = f"Create a simple {dietary_name} {meal_type} for {servings} servings using only: {', '.join([ing['name'] for ing in selected])}. Keep instructions short."
        instructions = await gemini_generate(prompt)
        fallback_instructions = f"Mix and cook {', '.join([ing['name'] for ing in selected])} for {'5' if 'snack' in input_lower else '10'} minutes."

        # Format ingredients
        ingredients_str = "; ".join([f"{ing['quantity_g']}g {ing['name']}" for ing in recipe_ingredients])

        recipe = {
            "solution": (
                f"Recipe: {dietary_name} {meal_type}\n"
                f"Ingredients: {ingredients_str}\n"
                f"Instructions: {instructions or fallback_instructions}\n"
                f"Servings: {servings}"
            ),
            "recipe_data": {
                "name": f"{dietary_name} {meal_type}",
                "ingredients": recipe_ingredients,
                "instructions": instructions or fallback_instructions,
                "servings": servings
            }
        }
        return recipe
    except Exception as e:
        return {"error": f"Error generating recipe: {str(e)}"}

# Check nutrition
def check_nutrition(recipe, user_input):
    try:
        if "error" in recipe:
            return f"Cannot verify: {recipe['error']}"
        recipe_data = recipe.get("recipe_data", {})
        if not recipe_data:
            return "Invalid recipe data"

        ingredients_db = load_ingredients()
        total_nutrients = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        for ing in recipe_data["ingredients"]:
            db_ing = next((i for i in ingredients_db if i["name"] == ing["name"]), None)
            if not db_ing:
                return f"Ingredient {ing['name']} not found in database."
            for nutrient in ["calories_per_100g", "protein_per_100g", "carbs_per_100g", "fat_per_100g"]:
                if db_ing.get(nutrient) is None:
                    return f"Missing {nutrient} for {ing['name']}"
            qty_g = ing["quantity_g"]
            total_nutrients["calories"] += (db_ing["calories_per_100g"] * qty_g / 100)
            total_nutrients["protein"] += (db_ing["protein_per_100g"] * qty_g / 100)
            total_nutrients["carbs"] += (db_ing["carbs_per_100g"] * qty_g / 100)
            total_nutrients["fat"] += (db_ing["fat_per_100g"] * qty_g / 100)

        per_serving = {k: v / recipe_data["servings"] for k, v in total_nutrients.items()}

        # Balance criteria
        if not (500 <= per_serving["calories"] <= 800):
            return f"Unbalanced: Calories ({per_serving['calories']:.1f}) not in 500–800 range. Reduce quantities."
        if per_serving["protein"] < 15:
            return f"Unbalanced: Protein ({per_serving['protein']:.1f}g) < 15g. Increase tofu or soy sauce."
        return (
            f"Recipe balanced: {per_serving['calories']:.1f} kcal, "
            f"{per_serving['protein']:.1f}g protein, {per_serving['carbs']:.1f}g carbs, "
            f"{per_serving['fat']:.1f}g fat per serving. TERMINATE"
        )
    except Exception as e:
        return f"Error checking nutrition: {str(e)}"

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
            "name": "generate_recipe",
            "description": "Generates a recipe based on user preferences",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "User's recipe preferences"},
                    "feedback": {"type": "string", "description": "Feedback from NutritionChecker", "default": None}
                },
                "required": ["user_input"]
            }
        },
        {
            "name": "check_nutrition",
            "description": "Checks nutritional balance of a recipe",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe": {"type": "object", "description": "Recipe to check"},
                    "user_input": {"type": "string", "description": "Original user input"}
                },
                "required": ["recipe", "user_input"]
            }
        }
    ]
}

manager_config = {
    "config_list": [
        {
            "model": "gemini-1.5-flash",
            "api_type": "google",
            "api_key": GOOGLE_API_KEY
        }
    ]
}

# Custom GroupChatManager
class CustomGroupChatManager(GroupChatManager):
    async def a_run_chat(self, messages=None, sender=None, config=None):
        groupchat = self._groupchat
        messages = messages or []
        if not isinstance(messages, list):
            messages = [messages]
        groupchat.messages.extend(messages)
        
        round_count = 0
        while round_count < groupchat.max_round:
            for msg in groupchat.messages[::-1]:
                if isinstance(msg, dict) and "content" in msg and "TERMINATE" in msg["content"].upper():
                    return msg

            speaker = groupchat.select_speaker(
                last_speaker=sender,
                selector=self
            )
            if not speaker:
                break

            reply = await speaker.a_generate_reply(
                messages=groupchat.messages,
                sender=self
            )

            if reply is None:
                break

            groupchat.append(reply, speaker)
            round_count += 1

        return groupchat.messages[-1] if groupchat.messages else None

# Initialize agents
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={
        "work_dir": "chef_tasks",
        "use_docker": False
    }
)

recipe_creator = AssistantAgent(
    name="RecipeCreator",
    llm_config=function_config,
    system_message=(
        "You are a chef. Use generate_recipe to create a recipe using only ingredients from ingredients.json. "
        "If NutritionChecker provides feedback, revise the recipe with generate_recipe, passing the feedback. "
        "Do not respond after a message containing 'TERMINATE'."
    ),
    function_map={"generate_recipe": generate_recipe}
)

nutrition_checker = AssistantAgent(
    name="NutritionChecker",
    llm_config=function_config,
    system_message=(
        "You are a nutritionist. Use check_nutrition to verify recipe balance (500–800 kcal, ≥15g protein per serving). "
        "If unbalanced, provide specific feedback. If balanced, output 'TERMINATE'. "
        "Do not respond after 'TERMINATE'."
    ),
    function_map={"check_nutrition": check_nutrition}
)

# Group chat
group_chat = GroupChat(
    agents=[recipe_creator, nutrition_checker],
    messages=[],
    max_round=10,
    speaker_selection_method="round_robin"
)

manager = CustomGroupChatManager(
    groupchat=group_chat,
    llm_config=manager_config
)

# Run chef
async def run_chef(preferences):
    try:
        await user_proxy.a_initiate_chat(
            recipient=manager,
            message=f"Generate a recipe: {preferences}"
        )
    except Exception as e:
        print(f"❌ Failed to generate recipe: {e}")

# Main loop
async def main():
    print("🚀 Personal Chef AI: Enter a recipe preference (e.g., 'vegan dinner for 2', 'quick vegan snack').")
    print("Type 'terminate' or 'exit' to quit.")
    
    while True:
        preferences = input("\nEnter preference: ").strip()
        
        if preferences.lower() in ["terminate", "exit"]:
            print("✅ Exiting Personal Chef AI.")
            break
        
        if not preferences:
            print("⚠️ Please enter a valid preference or 'terminate' to quit.")
            continue
        
        print(f"\nGenerating: {preferences}")
        await run_chef(preferences)

if __name__ == "__main__":
    asyncio.run(main())