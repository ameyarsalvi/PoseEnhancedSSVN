import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import os

# Input image path
input_path = "/home/asalvi/raw_imgs/test_image.png"  # Replace with your image path
save_dir = "/home/asalvi/raw_imgs/blur_outputs/"
os.makedirs(save_dir, exist_ok=True)

# Load image using OpenCV and convert to RGB
img = cv2.imread(input_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

# Define sigma values for Gaussian blur
sigma_values = [5, 10, 30, 60, 100]
blurred_images = []

# Generate and save blurred images
for sigma in sigma_values:
    transform = T.Compose([
        T.ToTensor(),
        T.GaussianBlur(kernel_size=15, sigma=sigma),
        T.ToPILImage()
    ])

    blurred_img = transform(img_pil)
    blurred_images.append(np.array(blurred_img))

    # Save to file
    save_path = os.path.join(save_dir, f"blur_sigma_{sigma}.png")
    blurred_img.save(save_path)
    print(f"Saved: {save_path}")
