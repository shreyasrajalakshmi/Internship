class BankAccount:
    def __init__(self, owner, balance=0):  
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount > 0: 
            self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if 0 < amount <= self.balance: 
            self.balance -= amount
        return self.balance

    def get_balance(self):
        return self.balance


def main():
    account = BankAccount(input("Enter your name: "), )  

    while True:
        choice = input("\n1. Deposit 2. Withdraw 3. Check Balance 4. Exit\nChoose: ")
        if choice == "1": 
            amount = float(input("Deposit amount: "))
            print(f"New Balance: {account.deposit(amount)}")
        elif choice == "2": 
            amount = float(input("Withdraw amount: "))
            print(f"New Balance: {account.withdraw(amount)}")
        elif choice == "3": 
            print(f"Balance: {account.get_balance()}")
        elif choice == "4": 
            print("Goodbye!")
            break
        else: 
            print("Invalid option.")

if __name__ == "__main__":  
    main()
