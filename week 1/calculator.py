def calculator():
    print("Simple Calculator: add, subtract, multiply, divide")

    try:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        op = input("Enter operation: ").strip().lower()

        operations = {
            "add": num1 + num2,
            "subtract": num1 - num2,
            "multiply": num1 * num2,
            "divide": "Error: Division by zero" if num2 == 0 else num1 / num2
        }

        result = operations.get(op, "Invalid operation")
        return f"Result: {result}" if isinstance(result, (int, float)) else result

    except ValueError:
        return "Invalid input"


print(calculator())

