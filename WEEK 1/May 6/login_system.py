#predefined username and password
correct_username = "user1"
correct_password = "12345"

#ask user to input username
username = input("Enter username:")

#check if username is correct
if username == correct_username:
    #ask for password only if username is correct
    password = input("Enter password:")

    #check if password is correct
    if password == correct_password:
        print("Login successful!")
    else:
        print("Incorrect password. Access denied.")
else:
    print("Username not found. Access denied.") #wrong username
