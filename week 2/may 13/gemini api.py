import google.generativeai as genai

def setup_gemini(api_key):
    """
    Configures the Gemini API with the provided API key.
    
    Args:
        api_key (str): Your Google Generative AI API key.
    
    Returns:
        genai.GenerativeModel: Configured Gemini model instance.
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro-latest')

def ask_gemini(model, prompt):
    """
    Sends a prompt to the Gemini model and retrieves the response.
    
    Args:
        model (GenerativeModel): Configured Gemini model.
        prompt (str): The text prompt to send.
    
    Returns:
        str: Generated response from the model.
    """
    response = model.generate_content(prompt)
    return response.text

# Main program
if __name__ == "__main__":
    # Replace this with your actual API key
    API_KEY = "AIzaSyCHy80eWH_N7Q9Xc0niq9OpxdNKaCoJmBQ"
    
    try:
        # Setup Gemini model
        model = setup_gemini(API_KEY)

        # Ask Gemini a question
        prompt = "Explain the main differences between Artificial Intelligence and Machine Learning."
        print("\n--- Gemini Output ---")
        response = ask_gemini(model, prompt)
        print(response)

    except Exception as e:
        print(f"An error occurred: {e}")
