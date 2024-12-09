import json
from transformers import AutoTokenizer

def analyze_response_lengths():
    # Load the dataset
    with open('logic_reasoning_dataset.json', 'r') as f:
        data = json.load(f)
    
    # Initialize tokenizer (using a common model)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Calculate token lengths of responses
    token_lengths = []
    for item in data:
        response = item['response']
        tokens = tokenizer(response)['input_ids']
        token_lengths.append(len(tokens))
    
    # Calculate average
    avg_length = sum(token_lengths) / len(token_lengths)
    
    print(f"Number of examples analyzed: {len(data)}")
    print(f"Average response length in tokens: {avg_length:.2f}")
    print(f"Min response length: {min(token_lengths)}")
    print(f"Max response length: {max(token_lengths)}")

if __name__ == "__main__":
    analyze_response_lengths()
