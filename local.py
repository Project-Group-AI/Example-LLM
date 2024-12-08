from transformers import AutoModelForCausalLM, AutoTokenizer

def use_mistral(prompt):
    """
    Use the Mistal 7B model for generate text.
    """
    try:
        # Load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype="auto",  # Use the optimal data type
            device_map="auto"    # Activate GPU or CPU usage automatically
        )
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate an answer
        outputs = model.generate(
            inputs.input_ids,
            max_length=300,  # Response length limit
            num_return_sequences=1,
            temperature=0.7,  # Creativity control
            top_k=50
        )
        
        # Decode and return response
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        return f"Error : {e}"

# Example of use
if __name__ == "__main__":
    prompt = "Can you explain me rules of chess?"
    reponse = use_mistral(prompt)
    print(f"Answer : {reponse}")
