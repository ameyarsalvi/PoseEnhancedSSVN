import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import seaborn as sns
from scipy.stats import gaussian_kde

# === CNN with configurable output dimension ===
class CustomCNN(nn.Module):
    def __init__(self, output_dim, input_channels=3):
        super(CustomCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 24 * 80, output_dim)
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

# === Get or load cached features ===
def get_or_load_features(image_dir, label, dim, cache_dir="activation_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{label}_{dim}.npy")

    if os.path.exists(cache_path):
        print(f"[Cache] Loading {cache_path}")
        return np.load(cache_path)

    print(f"[Compute] Extracting features: {label}, dim={dim}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = load_images(image_dir).to(device)

    model = CustomCNN(output_dim=dim).to(device).eval()
    outputs = []

    with torch.no_grad():
        for img in tqdm(images, desc=f"{label} - {dim}"):
            out = model(img[None]).squeeze(0).cpu().numpy()
            outputs.append(out)

    data = np.stack(outputs)
    np.save(cache_path, data)
    return data

# === Compute KDE with fixed support ===
def get_kde(data, support):
    kde = gaussian_kde(data)
    pdf = kde(support)
    return pdf / np.sum(pdf)  # Normalize to valid probability density

# === Main plotting function ===
def plot_distributions_grid(all_inputs_dict, dims, labels, colors,
                            save_path="activation_distribution_overlay",
                            data_save_dir="plot_data_cache"):

    os.makedirs(data_save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, len(dims), figsize=(4 * len(dims), 4), sharey=True)

    for i, dim in enumerate(dims):
        ax = axs[i]
        support = np.linspace(-3, 3, 1000)  # shared KDE support
        kde_data = {"support": support}

        for label, color in zip(labels, colors):
            inputs = all_inputs_dict[label][i]
            flat_vals = inputs.flatten()
            mean_val = np.mean(flat_vals)
            std_val = np.std(flat_vals)

            # Compute and store KDE
            pdf = get_kde(flat_vals, support)
            kde_data[label] = pdf

            # Plot KDE
            ax.plot(support, pdf, label=label, color=color, linewidth=2)
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=1)
            ax.fill_betweenx([0, ax.get_ylim()[1]], mean_val - std_val, mean_val + std_val,
                             color=color, alpha=0.1)

        ax.set_title(f"Dim = {dim}")
        ax.set_xlabel("Feature Value")
        if i == 0:
            ax.set_ylabel("Density")
        ax.grid(True)

        # Save KDE data for this dimension
        np.savez(os.path.join(data_save_dir, f"kde_data_dim{dim}.npz"), **kde_data)

    axs[0].legend(title="Dataset", fontsize=10)
    plt.suptitle("Activation Distributions for Varying Final Layer Sizes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.pdf", dpi=300)
    print(f"✅ Saved figure to {save_path}.png/.pdf and KDE data to {data_save_dir}/")
    plt.show()

# === Main pipeline ===
def main():
    dataset_paths = {
        'lanes':     '/home/asalvi/raw_vis_imgs/lanes/',
        'cones':     '/home/asalvi/raw_vis_imgs/cones/',
        'cylinders': '/home/asalvi/raw_vis_imgs/cylinders/',
    }

    output_dims = [2048, 1024, 512, 128, 64, 32, 16]
    all_inputs_dict = {label: [] for label in dataset_paths}
    colors = ['dodgerblue', 'forestgreen', 'darkorange']

    # Step 1: Load features for each dataset × dimension
    for label, path in dataset_paths.items():
        for dim in output_dims:
            features = get_or_load_features(path, label, dim)
            all_inputs_dict[label].append(features)

    # Step 2: Plot and save KDEs
    plot_distributions_grid(all_inputs_dict, output_dims, list(dataset_paths.keys()), colors)

if __name__ == '__main__':
    main()
