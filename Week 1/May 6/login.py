
valid_username = "shreyas"
valid_password = "131199"

def login():
    print("=== Simple Login System ===")

    username = input("Enter username: ").strip()
    if not username:
        print("Error: Username cannot be empty.")
        return

    password = input("Enter password: ").strip()
    if not password:
        print("Error: Password cannot be empty.")
        return

    if username != valid_username:
        print("Error: Username not found.")
    elif password != valid_password:
        print("Error: Incorrect password.")
    else:
        print("Login successful")


if __name__ == "__main__":
    login()

