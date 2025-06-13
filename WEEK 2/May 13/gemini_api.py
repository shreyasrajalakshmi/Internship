import google.generativeai as genai

api_key = 'AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg'

#configure Gemini API
genai.configure(api_key=api_key)

#initialize Gemini 1.5 Flash model
model = genai.GenerativeModel('models/gemini-1.5-flash')

#user input
user_prompt = input("Enter your prompt: ")

#generate content from user input
response = model.generate_content(user_prompt)

#display the result
print("\nGemini Response:")
print(response.text)
