from google import genai
from google.genai import types
import dotenv
from pathlib import Path

# environment variables

current_file_path = Path(__file__).resolve()
current_directory = current_file_path.parent
dotenv_path = current_directory / ".env"

dotenv.load_dotenv()
GEMINI_API_KEY = dotenv.get_key(dotenv_path, "GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
model = "gemini-2.5-flash"

def generate(prompt_and_context):
    response = client.models.generate_content(model=model, contents=prompt_and_context, config=config)
    return response.text