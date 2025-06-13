import re #regex

#open the file and read its content and converting to lowecase
with open("May 7\sample.txt", "r") as file: #filename or path
    text = file.read().lower()


#regex to extract only words
words = re.findall(r'\b[a-z]+\b', text)

#dictionary for counting
word_freq = {}

#count each word
for word in words:
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

#display the result
for word, freq in word_freq.items():
    print(f"{word}: {freq}")
