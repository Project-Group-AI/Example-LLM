import requests
from huggingface_hub import HfFolder

def mistral_via_api(prompt):
    """
    Uses the Hugging Face API to generate text with Mistral 7B Instruct.
    """
    # Set API URL
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    # Retrieve the token automatically saved by `huggingface-cli login`
    HF_TOKEN = HfFolder.get_token()
    if HF_TOKEN is None:
        return "Error : No tokens found. Make sure you are logged in via `huggingface-cli login`."

    # Set authentication header
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Prepare prompt data
    payload = {"inputs": prompt, "parameters": {"max_length": 300, "temperature": 0.7, "top_k": 50}}
    answer = requests.post(API_URL, headers=headers, json=payload)

    # Check answer
    if answer.status_code == 200:
        return answer.json()[0]["generated_text"]
    else:
        return f"Error : {answer.status_code} - {answer.json()}"

# Example of use
if __name__ == "__main__":
    prompt = "Can you explain me rules of chess?"
    answer = mistral_via_api(prompt)
    print(f"Answer : {answer}")
