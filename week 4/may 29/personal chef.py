import json
import asyncio
import random

# Default ingredient data (used if 'ingredients.json' is missing)
default_ingredients = {
    "tofu": {"calories": 144, "protein": 15, "carbs": 3, "fat": 9},
    "broccoli": {"calories": 55, "protein": 4, "carbs": 11, "fat": 1},
    "carrot": {"calories": 41, "protein": 1, "carbs": 10, "fat": 0},
    "bell pepper": {"calories": 30, "protein": 1, "carbs": 7, "fat": 0},
    "soy sauce": {"calories": 10, "protein": 1, "carbs": 1, "fat": 0},
    "garlic": {"calories": 5, "protein": 0, "carbs": 1, "fat": 0}
}

try:
    with open(r"D:\cusat\internship\Internship\week 4\may 29\ingredients.json") as f:
        INGREDIENTS_DB = json.load(f)
except FileNotFoundError:
    print("⚠️ Warning: 'ingredients.json' not found at the given path. Using default ingredient data.")
    INGREDIENTS_DB = default_ingredients

quotes = [
    "Turn your food into medicine and you won’t need medicines anymore.",
    "Don’t try to eat less. Try to eat right.",
    "The more you eat right, the chunkier your bank account gets.",
    "Drugs are not the answer to your disease. Nutrition is.",
    "Don’t stop eating plants.",
    "You might struggle more to change your diet than saying no to cigarettes.",
    "Health is wealth.",
    "Feel sick? Drink lemon water. Vegetables are always the answer to your diseases.",
    "Keep good food in your fridge to only eat good food.",
    "Heavier breakfast, lighter dinner.",
    "Healthy food, happy gut."
]

# Agent 1: Recipe Creator
def generate_recipe(preference: str, servings: int) -> dict:
    ingredients = [
        {"name": "tofu", "amount": "1.0"},
        {"name": "broccoli", "amount": "1.0"},
        {"name": "carrot", "amount": "1.0"},
        {"name": "bell pepper", "amount": "1.0"},
        {"name": "soy sauce", "amount": "1.0"},
        {"name": "garlic", "amount": "1.0"}
    ]

    instructions = [
        "Cut tofu and vegetables into bite-sized pieces.",
        "Stir-fry tofu in a hot pan until golden.",
        "Add garlic, then toss in all vegetables.",
        "Stir-fry with soy sauce until vegetables are tender.",
        "Serve hot."
    ]

    return {
        "title": f"{preference} - Stir-Fry Delight",
        "preference": preference,
        "servings": servings,
        "ingredients": ingredients,
        "instructions": instructions
    }

# Agent 2: Nutrition Checker
def check_nutrition(recipe: dict) -> (str, bool):
    total = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    for item in recipe["ingredients"]:
        name = item["name"]
        if name in INGREDIENTS_DB:
            for key in total:
                total[key] += INGREDIENTS_DB[name][key]
        else:
            print(f"⚠️ Ingredient '{name}' not found in nutrition database.")

    per_serving = {k: round(v / recipe["servings"], 1) for k, v in total.items()}

    balance = (
        per_serving["protein"] >= 7 and
        per_serving["calories"] <= 600 and
        per_serving["fat"] <= 20
    )

    summary = (
        f"Calories per serving: {per_serving['calories']} kcal\n"
        f"Protein: {per_serving['protein']} g\n"
        f"Carbs: {per_serving['carbs']} g\n"
        f"Fat: {per_serving['fat']} g\n"
    )

    return summary, balance

# Main loop
async def main():
    print("👨‍🍳 Personal Chef Assistant\nType 'exit' to quit.")

    eating_options = [
        "Vegan Dinner",
        "Quick Snack",
        "High-Protein Meal",
        "Low-Carb Lunch",
        "Comfort Food",
        "Gluten-Free Option",
        "Low-Fat Breakfast",
        "Spicy Delight"
    ]

    while True:
        print("\nWhat do you feel like eating?")
        for i, option in enumerate(eating_options, 1):
            print(f"{i}. {option}")

        choice = input("Enter the number of your choice (or 'exit' to quit): ")

        if choice.lower() == 'exit':
            break

        try:
            choice_num = int(choice)
            if not (1 <= choice_num <= len(eating_options)):
                print("❌ Please choose a valid number from the list.")
                continue
            preference = eating_options[choice_num - 1]
        except ValueError:
            print("❌ Please enter a valid number.")
            continue

        servings = input("How many servings? ")
        try:
            servings = int(servings)
        except ValueError:
            print("❌ Please enter a valid number for servings.")
            continue

        print(f"\n👨‍🍳 Recipe Creator: Here's your recipe for {preference.lower()}:\n")
        recipe = generate_recipe(preference, servings)

        print(f"🥗 Recipe: {recipe['title']}")
        print(f"Servings: {recipe['servings']}\n")
        print("Ingredients:")
        for item in recipe["ingredients"]:
            print(f"- {item['amount']} x {item['name']}")

        print("\nInstructions:")
        for step_num, step in enumerate(recipe["instructions"], start=1):
            print(f"{step_num}. {step}")

        print("\n🥦 Nutrition Checker:")
        nutrition_summary, is_balanced = check_nutrition(recipe)
        print(nutrition_summary)

        if is_balanced:
            print("✅ This recipe is nutritionally balanced.")
        else:
            print("⚠ Not Balanced.")

        print(f"\n💡 Quote of the day:\n\"{random.choice(quotes)}\"")

        print("\n✅ Chat complete. You can start again or type 'exit' to quit.\n")

if __name__ == "__main__":
    asyncio.run(main())
