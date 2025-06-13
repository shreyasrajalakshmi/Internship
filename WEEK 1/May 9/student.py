class StudentManager:
    students = []

    def __init__(self, name, class_name, place):
        self.name = name
        self.class_name = class_name
        self.place = place

    #add a new student
    @classmethod
    def add_student(cls, name, class_name, place):
        student = StudentManager(name, class_name, place)
        cls.students.append(student)
        print(f"Student '{name}' added successfully.")

    #remove a student by name
    @classmethod
    def remove_student(cls, name):
        for student in cls.students:
            if student.name == name:
                cls.students.remove(student)
                print(f"Student '{name}' removed successfully.")
                return
        print(f"Student '{name}' not found.")

    #search a student by name
    @classmethod
    def search_student(cls, name):
        for student in cls.students:
            if student.name == name:
                print("Student found:")
                student.display()
                return
        print("Student not found.")

    #display details of the student
    def display(self):
        print(f"Name: {self.name}")
        print(f"Class: {self.class_name}")
        print(f"Place: {self.place}")
        print("------")

    #display all students
    @classmethod
    def display_all(cls):
        if not cls.students:
            print("No student records to display.")
        else:
            print("\nAll Students:\n")
            for student in cls.students:
                student.display()


# --- Sample---

StudentManager.add_student("Alice", "10A", "Kochi")
StudentManager.add_student("Alex", "9B", "Trivandrum")
StudentManager.add_student("Chandler", "10A", "Kozhikode")

StudentManager.display_all()

StudentManager.search_student("Alex")
StudentManager.remove_student("Alex")
StudentManager.search_student("Alex")

print("\nAfter removal:")
StudentManager.display_all()
