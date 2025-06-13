def palindrome(given): #function definition

    if given == given[::-1]: #comparing string with it's reverse by [::-1]-slicing

        return f"'{given}' is a palindrome" #output if true
    else:

        return f"'{given}' is not a palindrome" #output if fals

user_input=input("Enter the string to check palindrome:") #taking input from user

print(palindrome(user_input)) #call the palindrome function with the user's input and print the result