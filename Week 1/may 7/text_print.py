import os

# Step 1: Ask the user for input
user_input = input("Enter text: ")

#__________________________________________________________#

# Specify the path where the file should be created
path = 'D:/cusat/internship/Internship/Week 1/may 7/user_input.txt'

#__________________________________________________________#

# Ensure the directory exists
os.makedirs(os.path.dirname(path), exist_ok=True)

#__________________________________________________________#

# Step 2: Save the input to the specified path
with open(path, 'w', encoding='utf-8') as file:
    file.write(f"Input file: {user_input}\n")
    
#__________________________________________________________#

# Step 3: Display the content of the file
with open(path, 'r', encoding='utf-8') as file:
    print("\nFile content:")
    print(file.read())
#__________________________________________________________#

print(f"\n File generated at: {path}")
