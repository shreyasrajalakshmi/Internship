
def is_palindrome(text):
  """
  Checks if a given string is a palindrome (reads the same forwards and backward).

  Args:
    text: The string to check.

  Returns:
    True if the string is a palindrome, False otherwise.
  """
  processed_text = ''.join(filter(str.isalnum, text)).lower()  # Remove non-alphanumeric characters and convert to lowercase
  return processed_text == processed_text[::-1]

# Example usage:
string_to_check = 'madam'
if is_palindrome(string_to_check):
  print(f"'{string_to_check}' is a palindrome.")
else:
  print(f"'{string_to_check}' is not a palindrome.")
