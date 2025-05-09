class Student:
    def __init__(self, student_id, name, age):  # Fixed constructor
        self.student_id = student_id
        self.name = name
        self.age = age

    def __str__(self):  # Fixed string representation
        return f"ID: {self.student_id}, Name: {self.name}, Age: {self.age}"


class StudentManagementSystem:
    def __init__(self):  # Fixed constructor
        self.students = {}

    def add_student(self, student_id, name, age):
        if student_id in self.students:
            print(f"Student with ID {student_id} already exists.")
        else:
            student = Student(student_id, name, age)
            self.students[student_id] = student
            print(f"Student {name} added successfully.")

    def remove_student(self, student_id):
        if student_id in self.students:
            removed_student = self.students.pop(student_id)
            print(f"Student {removed_student.name} removed successfully.")
        else:
            print(f"No student found with ID {student_id}.")

    def search_student(self, student_id):
        if student_id in self.students:
            print(self.students[student_id])
        else:
            print(f"No student found with ID {student_id}.")

    def display_all_students(self):
        if self.students:
            for student in self.students.values():
                print(student)
        else:
            print("No students available.")


def main():
    system = StudentManagementSystem()

    while True:
        print("\nStudent Management System")
        print("1. Add Student")
        print("2. Remove Student")
        print("3. Search Student")
        print("4. Display All Students")
        print("5. Exit")

        choice = input("Enter your choice (1,2,3,4,5): ")

        if choice == "1":
            student_id = input("Enter student ID: ")
            name = input("Enter student name: ")
            age = int(input("Enter student age: "))
            system.add_student(student_id, name, age)

        elif choice == "2":
            student_id = input("Enter student ID to remove: ")
            system.remove_student(student_id)

        elif choice == "3":
            student_id = input("Enter student ID to search: ")
            system.search_student(student_id)

        elif choice == "4":
            system.display_all_students()

        elif choice == "5":
            print("Exiting the system.")
            break

        else:
            print("Invalid choice. Please choose again.")


if __name__ == "__main__":  # Fixed block
    main()
