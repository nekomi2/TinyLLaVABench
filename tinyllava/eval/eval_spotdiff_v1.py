import argparse
import json
import torch
import os

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


def eval_model(args):
    disable_torch_init()

    # Load model components
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Read queries and image filenames from JSON
    with open(args.query_json, 'r') as f:
        query_data = json.load(f)

    results = []
    for item in query_data:
        # Prepare image file path
        image_file = os.path.join(args.image_dir, item['image'])
        image = load_image(image_file)
        images_tensor = process_images(
            [image], image_processor, model.config
        ).to(model.device, dtype=torch.float16)

        # Generate query prompt with image
        query = "What is the difference between these two images?"
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in query:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = image_token_se + "\n" + query

        # Prepare conversation
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # Generate output
        output_ids = model.generate(
            input_ids,
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

        # Collect output
        results.append({"id": item['id'], "response": output})

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
