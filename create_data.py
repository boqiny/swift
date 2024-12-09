import json
from typing import List, Dict

def create_training_dataset(json_data: List[Dict]) -> List[Dict]:
    """Create a training dataset from the provided json data."""
    
    # Process each example into the desired format
    processed_data = []
    for item in json_data:
        # Create the prompt text with numbered premises
        premises_text = ""
        for i, premise in enumerate(item['premises_nl'], 1):
            premises_text += f"({i}) {premise}\n"
            
        query = f"""Given a set of premises and a hypothesis, analyze whether the hypothesis logically follows from the premises.

###Premises: 
{premises_text}
###Hypothesis: 
{item['hypothesis_nl']}"""

        # Get the explanation and label
        response = item['explanation']
        
        processed_data.append({
            'query': query,
            'response': response
        })

    return processed_data

def main():
    # Load the JSON data
    with open('merged_logic_programs.json', 'r') as f:
        json_data = json.load(f)
    
    # Create the dataset
    dataset = create_training_dataset(json_data)
    
    # Save the dataset as JSON
    with open('logic_reasoning_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
