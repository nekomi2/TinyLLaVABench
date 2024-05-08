import argparse
import json
import torch
import os
from tqdm import tqdm

from tinyllava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
)
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO
import re

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def process_item(args, model, tokenizer, image_processor, item):
    # Load and process all images for the item
    image_files = [os.path.join(args.image_dir, img) for img in item['images']]
    images = [load_image(img_file) for img_file in image_files]
    images_tensor = process_images(
            images, image_processor, model.config
        ).to(model.device, dtype=torch.float16)
    # Adjust the prompt to include two image tokens
    query = "What is the difference between these two images?"
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN  # Two image tokens
    query = image_token_se + "\n" + query

    prompt = tokenizer_image_token(
        query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    output_ids = model.generate(
        prompt,
        images=images_tensor,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    output = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()
    print('out: ' + output)
    
    print("target: " + item['id'] + " " + item['conversations'][-1]['value'])
    return {"id": item['id'], "response": output}

def eval_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    with open(args.query_json, 'r') as f:
        query_data = json.load(f)

    results = []
    for item in tqdm(query_data, desc="Processing images", unit="image"):
        result = process_item(args, model, tokenizer, image_processor, item)
        results.append(result)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results to a JSON file
    with open(os.path.join(args.output_dir, 'output.json'), 'w') as file_out:
        json.dump(results, file_out, indent=4)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="bczhou/TinyLLaVA-1.5B")
    parser.add_argument("--query-json", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
