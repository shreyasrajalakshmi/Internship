void main() {
  // 1. Variables & Data Types
  String name = "Shreyas R";       // Name
  int age = 25;                    // Age
  double height = 173.9;           // Height in cm
  String region = "South India";   // Region
  String state = "Kerala";         // State
  String district = "Palakkad";    // District
  bool isLearning = true;          // true/false

  print("Name: $name");
  print("Age: $age years");
  print("Height: $height cm");
  print("Region: $region");
  print("State: $state");
  print("District: $district");
  print("Is Learning Flutter: $isLearning");

  // 2. final vs const
  final DateTime currentTime = DateTime.now(); // Runtime value
  
  print("Current Time: $currentTime");
} 