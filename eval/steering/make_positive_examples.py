import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time
import os
import argparse

# Set up the OpenAI API key
load_dotenv('.env', override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
arg_parser.add_argument("--neuron", type=int, help="The neuron id to generate examples for")
args = arg_parser.parse_args()

DEBUG = args.debug
num_examples = 25

# Define the prompt for the OpenAI API
prompt = """
Generate {num_examples} text examples that have the following feature: {feature_description}

Below are examples of text that have the feature described above.

Examples:
{examples}

Each text example should be around **twelve** words long and be unique. Try to be varied in the content of the examples.
"""

# Function to generate more examples using OpenAI API
def generate_examples(feature_description, examples, model="gpt-4o-mini", max_tokens=2000):
    # Construct messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates text examples."},
        {"role": "user", "content": prompt.format(feature_description=feature_description, examples=examples, num_examples=num_examples)}
    ]
    
    # Call the ChatCompletion API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=1,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()

# Example usage
feature_description = """
    the text examples all talk about **machine learning**. They MUST use the word "machine learning" in the text. Do not use the word "AI" or "artificial intelligence" in the text.
    Avoid using "machine learning" at the beginning of the text.
"""

examples = """
    1. "For modern natural language processing techniques, machine learning models are essential."
    2. "Advancements in machine learning have transformed how we process and understand text data."
    3. "The integration of machine learning in language translation systems has improved their accuracy significantly."
    4. "Feature extraction plays a critical role in many machine learning applications for text processing."
    5. "Supervised machine learning approaches often rely on labeled datasets for training."
    6. "Unsupervised machine learning methods enable clustering and topic modeling in large text corpora."
    7. "Preprocessing text data is a vital step for effective machine learning model performance."
    8. "In recent years, machine learning has been applied to sentiment analysis with great success."
    9. "Optimization algorithms are a backbone of machine learning in natural language processing."
    10. "The ability of machine learning to classify documents accurately depends on robust feature engineering."
"""

def extract_sentences(examples_text):
    # Split into lines and remove empty lines
    lines = [line.strip() for line in examples_text.split('\n') if line.strip()]
    
    # Find where numbered list starts (first line with digit followed by period)
    start_idx = 0
    for i, line in enumerate(lines):
        if line[0].isdigit() and line[1] == '.':
            start_idx = i
            break
    
    # Extract sentences from numbered lines
    new_sentences = []
    for line in lines[start_idx:]:
        if line[0].isdigit():
            # Remove number, period, quotes, and any ". \" artifacts
            sentence = line[2:].strip().strip('"').strip('. "\\')  # Added cleaning for ". \" artifacts
            new_sentences.append(sentence)
    
    return new_sentences

# Load existing sentences if file exists
json_path = f"eval/steering/datasets/latent_{args.neuron}.json"
all_sentences = []
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        all_sentences = json.load(f)["examples"]

os.makedirs("eval/steering/datasets", exist_ok=True)  # Ensure the directory exists


for _ in tqdm(range(16), desc="Generating examples") if not DEBUG else [0]:
    # sleep for 20 seconds to avoid rate limiting
    if not DEBUG:
        time.sleep(20)
    
    new_examples = generate_examples(feature_description, examples)
    new_sentences = extract_sentences(new_examples)
    
    # Add new sentences and remove duplicates while preserving order
    all_sentences.extend(new_sentences)
    all_sentences = list(dict.fromkeys(all_sentences))
    
    # Save to JSON after each batch
    output_data = {
        "examples": all_sentences
    }
    
    if not DEBUG:
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=4)
    
    print(f"Total unique sentences: {len(all_sentences)}")

