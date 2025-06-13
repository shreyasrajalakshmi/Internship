#input from the user
contents = input("Enter your contents: ")




#save the input to a file
with open("user_input.txt", "w") as file:
    file.write(contents)


#read the content back from the file
with open("user_input.txt", "r") as file:
    content = file.read()



#print the content read from the file
print("Content from the file:")
print(content)
