import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import seaborn as sns

# === CNN with configurable output dimension ===
class CustomCNN(nn.Module):
    def __init__(self, output_dim, input_channels=3):
        super(CustomCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 48, 160)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # -> (64, 24, 80)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),              # -> (64, 24, 80)
            nn.ReLU(),
            nn.Flatten(),                                                       # -> (122880,)
            nn.Linear(64 * 24 * 80, output_dim)                                 # final output_dim as param
        )

    def forward(self, x):
        return self.cnn(x)

# === Image loading and resizing ===
def load_images(image_dir):
    images = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            img_tensor = transforms.ToTensor()(img)
            images.append(img_tensor)
    return torch.stack(images)

# === Plot distribution of flattened outputs ===
def plot_distributions_grid(all_inputs, dims):
    fig, axs = plt.subplots(1, len(all_inputs), figsize=(4 * len(all_inputs), 4), sharey=True)

    for i, (inputs, dim) in enumerate(zip(all_inputs, dims)):
        flat_vals = inputs.flatten()
        mean_val = np.mean(flat_vals)
        std_val = np.std(flat_vals)

        ax = axs[i]
        sns.histplot(flat_vals, bins=100, kde=True, stat="density", color='skyblue', edgecolor='black', ax=ax)
        ax.axvline(mean_val, color='red', linestyle='--', label='Mean')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', label='-1σ')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', label='+1σ')
        ax.set_title(f"Dim = {dim}\nμ={mean_val:.2f}, σ={std_val:.2f}")
        ax.set_xlabel("Feature Value")
        if i == 0:
            ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.suptitle("Effect of Final Layer Dimension on Activation Distribution", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# === Main pipeline ===
def main(image_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading and resizing images...")
    images = load_images(image_dir).to(device)

    output_dims = [2048, 1024, 512, 128, 64, 32, 16]
    all_inputs = []

    for dim in output_dims:
        print(f"> Running CNN with final output dim = {dim}")
        model = CustomCNN(output_dim=dim).to(device).eval()

        inputs = []
        with torch.no_grad():
            for img in tqdm(images, desc=f"Extracting features ({dim})"):
                out = model(img[None]).squeeze(0).cpu().numpy()
                inputs.append(out)
        inputs = np.stack(inputs)  # (N_images, dim)
        all_inputs.append(inputs)

    # Plot all 7 distributions in one row
    plot_distributions_grid(all_inputs, output_dims)

if __name__ == '__main__':
    image_dir = '/home/asalvi/raw_vis_imgs/lanes/'  # 🔁 Update path as needed
    main(image_dir)
