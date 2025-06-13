class Bank:
    bank_name = 'SBI'

    def __init__(self, fname, lname, account_number):
        self.fname = fname
        self.lname = lname
        self.account_number = account_number
        self.balance = 1000 

    def deposit(self, amount):
        self.balance += amount
        print(f"Your {Bank.bank_name} account has been credited with {amount}.")
        print(f"Total balance: {self.balance}")

    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
            print(f"Your {Bank.bank_name} account has been debited with {amount}.")
            print(f"Total balance: {self.balance}")
        else:
            print("Insufficient balance.")

    def check_balance(self):
        print(f"Your current balance in {Bank.bank_name} is: {self.balance}")


per1 = Bank('Alex', 'Dunphy', 2350)


per1.deposit(1000)
per1.withdraw(300)
per1.check_balance()

per2 = Bank('Haley', 'Dunphy', 6789)
per2.check_balance()
