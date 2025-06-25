import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import seaborn as sns

# === Custom CNN (as per your screenshot) ===
class CustomCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(CustomCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 48, 160)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # -> (64, 24, 80)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),              # -> (64, 24, 80)
            nn.ReLU(),
            nn.Flatten(),                                                       # -> (122880,)
            nn.Linear(64 * 24 * 80, 64)                                          # Force final output to 64-dim
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
            
            # Resize to half: from 192x640 → 96x320
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            # Convert to tensor (C, H, W)
            img_tensor = transforms.ToTensor()(img)
            images.append(img_tensor)
    return torch.stack(images)

# --- Line Plot: Each image's 64-dim vector as a curve ---
def plot_feature_lines(inputs, max_images=100):
    plt.figure(figsize=(12, 6))
    for i in range(min(len(inputs), max_images)):
        plt.plot(range(inputs.shape[1]), inputs[i], alpha=0.3)
    plt.xlabel("Feature Index (0 to 63)")
    plt.ylabel("Activation Value")
    plt.title("CNN Output Vectors Across Images")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Heatmap: Image index vs. Feature index ---
def plot_feature_heatmap(inputs):
    plt.figure(figsize=(16, 8))
    sns.heatmap(inputs, cmap="viridis", cbar=True)
    plt.xlabel("Feature Index (0 to 63)")
    plt.ylabel("Image Index")
    plt.title("CNN Output Heatmap (Images vs Feature Activations)")
    plt.tight_layout()
    plt.show()

# --- Mean ± Std Dev per Feature ---
def plot_mean_std(inputs):
    mean_vals = np.mean(inputs, axis=0)
    std_vals = np.std(inputs, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(range(64), mean_vals, label="Mean Activation")
    plt.fill_between(range(64), mean_vals - std_vals, mean_vals + std_vals,
                     alpha=0.3, label="±1 Std Dev")
    plt.xlabel("Feature Index (0 to 63)")
    plt.ylabel("Activation Value")
    plt.title("Mean ± Std of CNN Feature Activations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_overall_distribution(inputs):
    flat_vals = inputs.flatten()
    mean_val = np.mean(flat_vals)
    std_val = np.std(flat_vals)

    plt.figure(figsize=(10, 5))
    sns.histplot(flat_vals, bins=100, kde=True, color='skyblue', stat="density", edgecolor='black')
    
    # Mean and ± std lines
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle=':', label=f'-1σ: {mean_val - std_val:.2f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle=':', label=f'+1σ: {mean_val + std_val:.2f}')

    plt.title("Distribution of All CNN Feature Values")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main pipeline ===
def main(image_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading and resizing images...")
    images = load_images(image_dir).to(device)

    model = CustomCNN().to(device).eval()

    with torch.no_grad():
        dummy = model(images[0][None])
        print(f"Output shape per image: {dummy.shape}")
        feature_dim = dummy.shape[1]

        inputs = []
        for img in tqdm(images):
            out = model(img[None]).squeeze(0).cpu().numpy()
            inputs.append(out)

    inputs = np.stack(inputs)  # Shape: (N_images, feature_dim)

    # --- Global histogram ---
    plt.figure(figsize=(12, 6))
    plt.hist(inputs.flatten(), bins=100, color='teal', edgecolor='black')
    plt.title('Histogram of All CNN Feature Values')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Additional plots ---
    plot_feature_lines(inputs)
    plot_feature_heatmap(inputs)
    plot_mean_std(inputs)
    plot_overall_distribution(inputs)


if __name__ == '__main__':
    image_dir = '/home/asalvi/raw_vis_imgs/cylinders/'  # 🔁 Change this path as needed
    main(image_dir)
