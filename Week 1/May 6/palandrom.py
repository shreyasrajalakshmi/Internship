import string

def is_palindrome(word):
    # Remove punctuation and convert to lowercase
    cleaned = ''.join(char.lower() for char in word if char.isalnum())
    return cleaned == cleaned[::-1]

def main():
    word = input("Enter a word or phrase: ")
    if is_palindrome(word):
        print(f'"{word}" is a palindrome!')
    else:
        print(f'"{word}" is not a palindrome.')

if __name__ == "__main__":
    main()
