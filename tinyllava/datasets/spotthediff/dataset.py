import random
import gdown
import zipfile
import os
import tqdm
import json


# Define the URL and output file
current_file_dir = os.path.dirname(__file__)
url = 'https://drive.google.com/uc?id=1OVb4_3Uec_xbyUk90aWC6LFpKsIOtR7v'
output = 'resized_images.zip'
target_directory = os.path.join(current_file_dir, 'images')

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
            sample_dict = {
                'id': sample['img_id'],
                'image': f'{sample["img_id"]}.png'
            }
            sample_dict['conversations'] = [
                {"from": "human", "value": "<image>\n" + random.choice(description_list)},
                {"from": "gpt", "value": '\n'.join(sample['sentences'])}
            ]
            if sample_dict['conversations'][1]['value'] == "":
                sample_dict['conversations'][1]['value'] = 'There are no differences between the images.'
                
            spotdiff_data.append(sample_dict)

    # Dump data to a JSON file
    with open(output_path, 'w') as f:
        json.dump(spotdiff_data, f, indent=4)
        print(f"SpotTheDiff dataset saved to {output_path}")

# Process each dataset
if __name__ == '__main__':
    # Check if the directory exists and has files
    if not os.path.exists(target_directory) or not os.listdir(f'{target_directory}/resized_images'):
        # Download the file
        gdown.download(url, output, quiet=False)
        print('saving images...')
        # Unzip the file into the 'images' directory
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(target_directory)
        print('Images saved successfully.')
        # Optionally, remove the zip file after extraction
        os.remove(output)
    else:
        print(f"Directory '{target_directory}' already exists and is not empty.")

    process_data(train_data_path, os.path.join(current_file_dir, 'spotthediff_train.json'))
    process_data(val_data_path, os.path.join(current_file_dir, 'spotthediff_val.json'))
    process_data(test_data_path, os.path.join(current_file_dir, 'spotthediff_test.json'))
