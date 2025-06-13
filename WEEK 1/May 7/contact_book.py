file_name = r'./May 7/contacts.txt' 

#functions to load from file
def load_contacts():
    contacts={}
    
    with open(file_name,'r') as file:
        for i in file:
            if ',' in i:
                name, phone = i.strip().split(',',1)
                contacts[name.strip()] = phone.strip()
    return contacts

#save from dictionary to file 
def save_contacts(contacts):
    with open(file_name,'a') as file:
        for name,phone in contacts.items():
            file.write(f"{name},{phone}\n")
            
    print('saved.')

#add new contact to dictionary
def add_contact(contacts):
    name=input('Enter name:').strip()
    phone=input('Enter phone number:').strip()
    contacts[name]=phone
    print(f"contact '{name}' added.")

#display contacts
def view_contacts(contacts):
    if not contacts:
        print('No contacts found.')
    else:
        print('\nContact list:\n')
        for name,phone in contacts.items():
            print(f"{name}: {phone}")

#menu

def main():
    
    contacts=load_contacts()

    while True:
        print("\n=== Contact Book ===")
        print("1. Add Contact")
        print("2. View Contacts")
        print("3. Save and Exit")

        choice = input("Choose an option (1/2/3): ").strip()

        if choice == "1":
            add_contact(contacts)
        elif choice == "2":
            view_contacts(contacts)
        elif choice == "3":
            save_contacts(contacts)
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()

