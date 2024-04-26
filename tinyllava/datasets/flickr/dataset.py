import shortuuid
from datasets import load_dataset
from PIL import Image
import random
import json
import tqdm
import os

# Load the Flickr30k dataset
ds = load_dataset("nlphuji/flickr30k")

# Paths for storing images and JSON data
flickr_image_path = 'datasets/flickr/image'
flickr_data_path = 'datasets/flickr/flickr30k_captions.json'

# Description prompts (the same as provided by you)
description_list = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented."
]

flickr_data = []

# Process the dataset
for sample in tqdm.tqdm(ds['test']):
    uuid = shortuuid.uuid()
    sample_dict = {
        'id': uuid,
        'image': f'flickr30k/image/{uuid}.jpg'
    }
    
    # Save the image to a specified path
    image = sample['image']
    image.save(os.path.join(flickr_image_path, f'{uuid}.jpg'))
    
    # Create conversation based on random description and the provided captions
    conversations = [
        {"from": "human", "value": "<image>\n" + random.choice(description_list)},
        {"from": "gpt", "value": sample['caption']}
    ]
    sample_dict['conversations'] = conversations
    flickr_data.append(sample_dict)

# Dump data to a JSON file
with open(flickr_data_path, 'w') as f:
    json.dump(flickr_data, f, indent=4)
