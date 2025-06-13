import google.generativeai as genai

genai.configure(api_key="AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")

for model in genai.list_models():
    print(model.name)
