void main() {
  // Step 1: Create a list of hobbies
  List<String> hobbies = ["Drawing", "Swimming", "Travelling", "Photography", "Coding"];

  // Step 2: Print hobbies using a function with a for loop
  printHobbiesForLoop(hobbies);

  // Step 3: Print hobbies using a function with a while loop
  printHobbiesWhileLoop(hobbies);
}

// Function to print hobbies using a for loop
void printHobbiesForLoop(List<String> list) {
  print(" Favorite Hobbies (For Loop):");
  for (int i = 0; i < list.length; i++) {
    print("Hobby $i: ${list[i]}");
  }
}

// Function to print hobbies using a while loop
void printHobbiesWhileLoop(List<String> list) {
  print("\n Favorite Hobbies (While Loop):");
  int index = 0;
  while (index < list.length) {
    print("Hobby $index: ${list[index]}");
    index++;
  }
}
