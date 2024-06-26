import json
import os
import random
import tqdm

# Define the paths for data storage and descriptions
current_file_dir = os.path.dirname(__file__)
train_data_path = os.path.join(current_file_dir, 'spot-the-diff/data/annotations/train.json')
val_data_path = os.path.join(current_file_dir, 'spot-the-diff/data/annotations/val.json')
test_data_path = os.path.join(current_file_dir, 'spot-the-diff/data/annotations/test.json')
output_path = os.path.join(current_file_dir, 'spotthediff_train_v6.json')

# Description prompts
description_list = [
    "Describe the difference between the two images.",
    # Add other descriptions as needed
]

def process_data(data_path, output_path):
    print(f'Generating JSON for {output_path}...')
    with open(data_path, 'r') as f:
        data = json.load(f)
        spotdiff_data = []
        for sample in tqdm.tqdm(data):
            img_base = sample['img_id']
            sample_dict = {
                'id': img_base,
                'images': [f'{img_base}.png', f'{img_base}_2.png']
            }
            sample_dict['conversations'] = [
                {"from": "human", "value": "<image>\n<image>\n" + random.choice(description_list)},
                {"from": "gpt", "value": ('\n'.join(sample['sentences']) + '\n') if sample['sentences'][0] != "" else 'There are no differences between the images.\n'}
            ]
            spotdiff_data.append(sample_dict)

    # Save the modified dataset to a JSON file
    with open(output_path, 'w') as f:
        json.dump(spotdiff_data, f, indent=4)
        print(f"SpotTheDiff dataset saved to {output_path}")

if __name__ == '__main__':
    process_data(train_data_path, output_path)
    process_data(val_data_path, os.path.join(current_file_dir, 'spotthediff_val_v6.json'))
    process_data(test_data_path, os.path.join(current_file_dir, 'spotthediff_test_v6.json'))
