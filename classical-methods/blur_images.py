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
kernal_values = [0.001, 5, 15, 35, 55]
blurred_images = []

blur = {
        'kernal' : [3, 15, 25, 35, 45],
        'sigma' : [0.001, 5, 15, 35, 55]
    }

# Generate and save blurred images
#for kernal in kernal_values:
#    for sigma in sigma_values:
for sigma_val, kernal_val in zip(blur['sigma'], blur['kernal']):  
    transform = T.Compose([
        T.ToTensor(),
        T.GaussianBlur(kernel_size=kernal_val, sigma=sigma_val),
        T.ToPILImage()
    ])

    blurred_img = transform(img_pil)
    blurred_images.append(np.array(blurred_img))

    # Save to file
    save_path = os.path.join(save_dir, f"blur_{kernal_val}_{sigma_val}.png")
    blurred_img.save(save_path)
    print(f"Saved: {save_path}")
