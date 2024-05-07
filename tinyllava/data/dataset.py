import torch
from PIL import Image
import os
import copy

def process_image(image_file, image_folder, processor, image_aspect_ratio):
    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
    if image_aspect_ratio == 'pad':
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
    return processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    sources = self.list_data_dict[i]
    if isinstance(i, int):
        sources = [sources]
    
    images = []
    if 'image' in sources[0]:
        image_data = sources[0]['image']
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        
        if isinstance(image_data, list):
            for image_file in image_data:
                image_tensor = process_image(image_file, image_folder, processor, self.data_args.image_aspect_ratio)
                images.append(image_tensor)
        else:
            image_tensor = process_image(image_data, image_folder, processor, self.data_args.image_aspect_ratio)
            images.append(image_tensor)

        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.data_args)
    else:
        sources = copy.deepcopy([e["conversations"] for e in sources])
    
    data_dict = preprocess(
        sources,
        self.tokenizer,
        has_image=('image' in sources[0]))

    if isinstance(i, int):
        for key in data_dict:
            data_dict[key] = data_dict[key][0]

    # Process and include images in the data dict
    if images:
        data_dict['image'] = torch.stack(images) if len(images) > 1 else images[0]
    elif self.data_args.is_multimodal:
        # No images are present, but the model is multimodal
        crop_size = self.data_args.image_processor.crop_size
        num_images = len(image_data) if isinstance(image_data, list) else 1
        data_dict['image'] = torch.zeros((num_images, 3, crop_size['height'], crop_size['width']))
    return data_dict
