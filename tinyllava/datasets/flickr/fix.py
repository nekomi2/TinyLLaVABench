import json

# Path to the original JSON file
input_json_path = 'datasets/flickr/flickr30k_captions.json'
# Path to the output JSON file
output_json_path = 'datasets/flickr/flickr30k_captions.json'

# Load the data from the JSON file
with open(input_json_path, 'r') as file:
    data = json.load(file)

# Modify the data
for item in data:
    for conversation in item['conversations']:
        if conversation['from'] == 'gpt':
            # Update the value to only the first caption
            if isinstance(conversation['value'], list):
                conversation['value'] = conversation['value'][0]

# Save the modified data to a new JSON file
with open(output_json_path, 'w') as file:
    json.dump(data, file, indent=4)

print("JSON data has been updated and saved.")
