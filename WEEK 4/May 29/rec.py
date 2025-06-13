import os
import json
import asyncio
import random
from autogen import AssistantAgent, UserProxyAgent
import google.generativeai as genai
from autogen.agentchat import GroupChat, GroupChatManager

# Gemini API Setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")  # Replace with your Google API key or set GEMINI_API_KEY environment variable
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise Exception(f"Failed to configure Gemini API: {str(e)}")

# Gemini Retry Mechanism
async def gemini_generate(prompt: str, model_name="gemini-1.5-flash", retries: int = 5, delay: int = 5) -> str:
    """Generate content using Gemini API with retry logic."""
    model = genai.GenerativeModel(model_name)
    for i in range(retries):
        try:
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(prompt))
            return response.text
        except genai.types.generation_types.BlockedPromptException:
            return "ERROR: Prompt blocked by Gemini API."
        except genai.types.generation_types.StopCandidateException:
            return "ERROR: Generation stopped by Gemini API."
        except Exception as e:
            if "429" in str(e):
                return "ERROR: Gemini API quota exceeded."
            await asyncio.sleep(delay)
    return "ERROR: Gemini API failed after retries."

# Load ingredients from JSON
def load_ingredients():
    try:
        with open("ingredients.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return []

# Generate recipe
async def generate_recipe(user_input: str) -> dict:
    """Generate a recipe based on user preferences."""
    try:
        ingredients_db = load_ingredients()
        if not ingredients_db:
            return {"error": "Ingredients database empty or not found"}

        # Parse user input (e.g., "vegan dinner for 2 people")
        input_lower = user_input.lower()
        dietary = "vegan" if "vegan" in input_lower else ""
        servings = 2
        if "for" in input_lower:
            try:
                servings = int(input_lower.split("for")[1].split()[0])
            except (IndexError, ValueError):
                pass

        # Filter ingredients by dietary preference
        valid_ingredients = [
            ing for ing in ingredients_db if not dietary or dietary in [d.lower() for d in ing["dietary"]]
        ]
        if not valid_ingredients:
            return {"error": "No ingredients match dietary preference"}

        # Select ingredients (random 3 for demo)
        selected = random.sample(valid_ingredients, min(3, len(valid_ingredients)))
        recipe_ingredients = [
            {"name": ing["name"], "quantity_g": random.randint(100, 200)} for ing in selected
        ]

        # Generate instructions using Gemini
        dietary_name = dietary.capitalize() or "Custom"
        prompt = f"Create a {dietary_name} recipe for {servings} servings using ingredients: {', '.join([ing['name'] for ing in selected])}."
        instructions = await gemini_generate(prompt)

        recipe = {
            "solution": f"Recipe: {dietary_name} Recipe\nIngredients: " + "; ".join(
                [f"{ing['quantity_g']}g {ing['name']}" for ing in recipe_ingredients]
            ) + f"\nInstructions: {instructions}\nServings: {servings}",
            "recipe_data": {
                "name": f"{dietary_name} Recipe",
                "ingredients": recipe_ingredients,
                "instructions": instructions or f"Mix and cook {', '.join([ing['name'] for ing in selected])} for 10 minutes.",
                "servings": servings
            }
        }
        return recipe
    except Exception as e:
        return {"error": f"Error generating recipe: {str(e)}"}

# Check nutrition
def check_nutrition(recipe, user_input):
    """Check nutritional balance of a recipe."""
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
                return f"Ingredient {ing['name']} not found in database"
            qty_g = ing["quantity_g"]
            total_nutrients["calories"] += (db_ing["calories_per_100g"] * qty_g / 100)
            total_nutrients["protein"] += (db_ing["protein_per_100g"] * qty_g / 100)
            total_nutrients["carbs"] += (db_ing["carbs_per_100g"] * qty_g / 100)
            total_nutrients["fat"] += (db_ing["fat_per_100g"] * qty_g / 100)

        per_serving = {k: v / recipe_data["servings"] for k, v in total_nutrients.items()}

        # Balance criteria
        if not (500 <= per_serving["calories"] <= 800):
            return f"Unbalanced: Calories per serving ({per_serving['calories']:.1f}) not in 500–800 range."
        if per_serving["protein"] < 15:
            return f"Unbalanced: Protein per serving ({per_serving['protein']:.1f}g) < 15g."
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
                    "user_input": {"type": "string", "description": "User's recipe preferences"}
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
                break

            reply = await speaker.a_generate_reply(
                messages=groupchat.messages,
                sender=self
            )

            if reply is None:
                break

            groupchat.append(reply, speaker)

            if isinstance(reply, dict) and "content" in reply:
                if "TERMINATE" in reply["content"].upper():
                    break

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
    system_message="You are a chef. Use generate_recipe to create a recipe based on user preferences. Revise if NutritionChecker suggests changes.",
    function_map={"generate_recipe": generate_recipe}
)

nutrition_checker = AssistantAgent(
    name="NutritionChecker",
    llm_config=function_config,
    system_message="You are a nutritionist. Use check_nutrition to verify recipe balance. Output 'TERMINATE' if balanced, or suggest fixes if not.",
    function_map={"check_nutrition": check_nutrition}
)

# Group chat (round-robin)
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