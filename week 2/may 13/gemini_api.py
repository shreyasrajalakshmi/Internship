# --------------Import Gemini API --------------
import google.generativeai as genai

# -------------- Set API Key --------------
api_key = "AIzaSyCHy80eWH_N7Q9Xc0niq9OpxdNKaCoJmBQ"  
genai.configure(api_key=api_key)

#  --------------Load Gemini model --------------
model = genai.GenerativeModel("gemini-1.5-flash")

# # --------------1 Text Generation --------------
def generate_text(prompt):
    """Generates text based on the given prompt."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# #2  --------------Text Summarization --------------
def text_summarization(text):
    """Summarizes the given text."""
    try:
        response = model.generate_content(f"Summarize this: {text}")
        return response.text
    except Exception as e:
        return f"Error: {e}"

# #3  --------------Question Answering --------------
def question_answering(context, question):
    """Answers a question based on the given context."""
    try:
        response = model.generate_content(f"Question: {question} Context: {context}")
        return response.text
    except Exception as e:
        return f"Error: {e}"

# #4  --------------Sentiment Analysis --------------
def sentiment_analysis(text):
    """Analyzes the sentiment of the given text."""
    try:
        response = model.generate_content(f"Analyze the sentiment of this text: {text}")
        return response.text
    except Exception as e:
        return f"Error: {e}"

# #5  --------------Text Translation --------------
def text_translation(text, target_language):
    """Translates the given text to the target language."""
    try:
        response = model.generate_content(f"Translate this text to {target_language}: {text}")
        return response.text
    except Exception as e:
        return f"Error: {e}"

#  --------------Example Usage --------------
if __name__ == "__main__":
    # 1. Text generation
    print("**Text Generation:**")
    print(generate_text("The quick brown fox"))
    
    # 2. Summarization
    print("\n**Summarization:**")
    print(text_summarization("The quick brown fox jumps over the lazy dog"))
    
    # 3. Question answering
    print("\n**Question Answering:**")
    print(question_answering("The quick brown fox jumps over the lazy dog", "What does the fox jump over?"))
    
    # 4. Sentiment analysis
    print("\n**Sentiment Analysis:**")
    print(sentiment_analysis("The quick brown fox jumps over the lazy dog"))
    
    # 5. Translation
    print("\n**Translation (English to Spanish):**")
    print(text_translation("The quick brown fox jumps over the lazy dog", "Spanish"))
