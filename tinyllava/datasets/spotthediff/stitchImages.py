import json
from PIL import Image
import os
import tqdm

current_file_dir = os.path.dirname(__file__)


def stitch_images(
    image_id,
    source_folder=os.path.join(current_file_dir, "images/resized_images"),
    target_folder=os.path.join(current_file_dir, "images/stitched"),
):
    img1_path = os.path.join(source_folder, f"{image_id}.png")
    img2_path = os.path.join(source_folder, f"{image_id}_2.png")

    # Open the images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Get the maximum height of the two images
    max_height = max(img1.height, img2.height)

    # Create a new image with the combined width and the maximum height
    total_width = img1.width + img2.width
    new_img = Image.new("RGB", (total_width, max_height))

    # Paste the images into the new image
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    # Save the new image
    new_img.save(os.path.join(target_folder, f"{image_id}.png"))

def stitch_images_for_dataset(data_path):
        if not os.path.exists(os.path.join(current_file_dir, "images/stitched")):
            os.makedirs(os.path.join(current_file_dir, "images/stitched"))
        with open(os.path.join(current_file_dir, data_path), "r") as f:
            train_data = json.load(f)
            image_ids = [sample["id"] for sample in train_data]
        for image_id in tqdm.tqdm(image_ids):
            stitch_images(image_id)

if __name__ == "__main__":
    image_ids = []
    train_data_path = os.path.join(current_file_dir, "spotthediff_train.json")
    val_data_path = os.path.join(current_file_dir, "spotthediff_val.json")
    test_data_path = os.path.join(current_file_dir, "spotthediff_test.json")
    stitch_images_for_dataset(train_data_path)
    stitch_images_for_dataset(val_data_path)
    stitch_images_for_dataset(test_data_path)
    
