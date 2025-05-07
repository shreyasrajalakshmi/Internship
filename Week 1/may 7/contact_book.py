import json

# Load contacts from a file
def load_contacts(filename='contacts.json'):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
#----------------------------------------------------------------#

# Save contacts to a file
def save_contacts(contacts, filename='contacts.json'):
    with open(filename, 'w') as file:
        json.dump(contacts, file, indent=4)
        
#----------------------------------------------------------------#        

#  Add a contact 
def add_contact(contacts):
    name = input("Enter name: ")
    phone = input("Enter phone: ")
    email = input("Enter email: ")
    contacts[name] = {'phone': phone, 'email': email}
    print(f"✅ {name} added successfully!")

#----------------------------------------------------------------#

# Display all contacts
def display_contacts(contacts):
    if not contacts:
        print("No contacts found.")
    else:
        print("\n****Contact List****")
        for name, info in contacts.items():
            print(f"{name}: {info['phone']}, {info['email']}")
        print("______________________________________________________________________")
        
#----------------------------------------------------------------#

#  Main program 
def main():
    contacts = load_contacts()

    while True:
        choice = input("\n1. Add  2. Show  3. Save & Exit: ")
        if choice == '1': 
            add_contact(contacts)
        elif choice == '2': 
            display_contacts(contacts)
        elif choice == '3': 
            save_contacts(contacts)
            print("Contacts saved. Exiting...")
            break
        else:
            print("❌ Invalid choice. Please try again.")
            
#----------------------------------------------------------------#            

# Start the program correctly
if __name__ == '__main__':
    main()
