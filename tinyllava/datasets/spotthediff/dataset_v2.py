import random
import gdown
import zipfile
import os
import tqdm
import json


# Define the URL and output file
current_file_dir = os.path.dirname(__file__)
# Paths for storing images and JSON data
spotdiff_image_path = os.path.join(current_file_dir, 'images')
spotdiff_data_path = os.path.join(current_file_dir, 'spotthediff.json')
train_data_path = os.path.join(current_file_dir, 'spot-the-diff/data/annotations/train.json')
val_data_path = os.path.join(current_file_dir, 'spot-the-diff/data/annotations/val.json')
test_data_path = os.path.join(current_file_dir, 'spot-the-diff/data/annotations/test.json')

# Description prompts (the same as provided by you)
description_list = [
    "Describe the difference between the two images.",
    "Identify the key distinctions between the two pictures.",
    "Explain the main contrasts between these images.",
    "Highlight the differences in the visual elements of the two images.",
    "Summarize the distinct features present in each image.",
    "Detail the variations observed between the two pictures.",
    "Outline the primary differences in the imagery presented.",
    "Discuss the contrasting elements found in each image.",
    "Provide a comparative analysis of the two images.",
    "Explain how the two images differ from each other.",
    "Describe what sets these two images apart."
]

def process_data(data_path, output_path):
    print(f'generating json for {output_path}...')
    with open(data_path, 'r') as f:
        data = json.load(f)
        spotdiff_data = []
        for sample in tqdm.tqdm(data):
            for sentence in sample['sentences']:
                sample_dict = {
                    'id': sample['img_id'],
                    'image': f'{sample["img_id"]}.png'
                }
                sample_dict['conversations'] = [
                    {"from": "human", "value": "<image>\n" + random.choice(description_list)},
                    {"from": "gpt", "value": sentence if sentence.strip() != "" else 'There are no differences between the images.'}
                ]
                spotdiff_data.append(sample_dict)

    # Dump data to a JSON file
    with open(output_path, 'w') as f:
        json.dump(spotdiff_data, f, indent=4)
        print(f"SpotTheDiff dataset saved to {output_path}")

# Process each dataset
if __name__ == '__main__':
    process_data(train_data_path, os.path.join(current_file_dir, 'spotthediff_train_v2.json'))
    # process_data(val_data_path, os.path.join(current_file_dir, 'spotthediff_val.json'))
    # process_data(test_data_path, os.path.join(current_file_dir, 'spotthediff_test.json'))
