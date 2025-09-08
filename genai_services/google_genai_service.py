import json
import re

from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def google_genai_response(query):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=query
        )
        json_match = re.search(r'```json(.*?)```', response.text, re.DOTALL)
        if json_match:
            json_string = json_match.group(1).strip()
            data = json.loads(json_string)
            return data
        else:
            print("No JSON content found in the response.")
            return None

    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return None