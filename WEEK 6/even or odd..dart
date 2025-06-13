void main() {
  // Greeting the user
  greetUser("Shreyas R");

  // Check if a number is even or odd
  int myNumber = 17;
  String result = checkEvenOdd(myNumber);
  print("Result: $result");
}

// Function to greet the user
void greetUser(String name) {
  print("Hello, $name! Welcome to Flutter.");
}

// Function to check if a number is even or odd
String checkEvenOdd(int number) {
  if (number % 2 == 0) {
    return "$number is even.";
  } else {
    return "$number is odd.";
  }
}
