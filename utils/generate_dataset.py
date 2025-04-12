from stegano import lsb
from PIL import Image
import os

def generate_dataset(input_folder, output_cover, output_stego, message="Secret"):
    os.makedirs(output_cover, exist_ok=True)
    os.makedirs(output_stego, exist_ok=True)

    for idx, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(os.path.join(input_folder, filename))
            img = img.resize((128, 128))

            cover_path = os.path.join(output_cover, f"{idx}.png")
            stego_path = os.path.join(output_stego, f"{idx}.png")

            img.save(cover_path)
            stego = lsb.hide(cover_path, message)
            stego.save(stego_path)

# Usage
generate_dataset("cover_images", "dataset/cover", "dataset/stego")
