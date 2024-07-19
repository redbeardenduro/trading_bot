import os
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_code_suggestions(code):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Here is some Python code:\n{code}\nWhat changes or improvements can be made?",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    suggestions = response.choices[0].text.strip()
    return suggestions

def read_code_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    return code

def main():
    code_files = ["path/to/your/code.py"]  # Replace with your actual file paths
    for code_file_path in code_files:
        code = read_code_file(code_file_path)
        suggestions = get_code_suggestions(code)
        print(f"Suggestions for {code_file_path}:\n{suggestions}\n")

if __name__ == "__main__":
    main()
