import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.stats import gaussian_kde

# === Settings ===
output_dims = [2048, 1024, 512, 128, 64, 32, 16]
cache_dir = "activation_cache"
datasets = ['cones', 'lanes', 'cylinders']
colors = {'cones': 'dodgerblue', 'lanes': 'forestgreen', 'cylinders': 'darkorange'}

# LaTeX-style font and font size
plt.rcParams.update({
    "font.size": 14,
    "text.usetex": False,
    "font.family": "serif"
})

# === Helper: Smooth and normalize KDE ===
def get_kde_probs(data, support):
    kde = gaussian_kde(data)
    pdf = kde(support)
    pdf /= np.sum(pdf)  # normalize to make it a valid distribution
    return pdf

# === KL Divergence Plotting + Summary Bar Plot ===
def plot_and_kl(max_samples=1000):
    fig, axs = plt.subplots(2, 4, figsize=(22, 8), sharey=False)
    support = np.linspace(-3, 3, 1000)
    epsilon = 1e-10

    # Store KL values
    kl_values_lanes = []
    kl_values_cylinders = []

    for i, dim in enumerate(output_dims):
        ax = axs[i // 4, i % 4]
        activations = {}

        # === Load and subsample ===
        for dset in datasets:
            file = os.path.join(cache_dir, f"{dset}_{dim}.npy")
            full_data = np.load(file)
            N = min(max_samples, full_data.shape[0])
            indices = np.random.choice(full_data.shape[0], size=N, replace=False)
            data = full_data[indices].flatten()
            activations[dset] = data

        # === Plot KDEs ===
        for dset in datasets:
            sns.kdeplot(activations[dset], ax=ax, label=dset, color=colors[dset], linewidth=2)

        ax.set_title(f"Dim = {dim}")
        ax.set_xlabel("Feature Value")
        if i % 4 == 0:
            ax.set_ylabel("Density")
        ax.grid(True)

        # === KL Divergence Calculation ===
        p = get_kde_probs(activations['cones'], support)
        for dset in ['lanes', 'cylinders']:
            q = get_kde_probs(activations[dset], support)
            p_clipped = np.clip(p, epsilon, 1)
            q_clipped = np.clip(q, epsilon, 1)
            kl_kde = entropy(p_clipped, q_clipped)

            if np.isinf(kl_kde) or np.isnan(kl_kde):
                mu_p, var_p = np.mean(activations['cones']), np.var(activations['cones'])
                mu_q, var_q = np.mean(activations[dset]), np.var(activations[dset])
                kl_kde = np.log(np.sqrt(var_q) / np.sqrt(var_p)) + \
                         (var_p + (mu_p - mu_q)**2) / (2 * var_q) - 0.5

            if dset == 'lanes':
                kl_values_lanes.append(kl_kde)
            else:
                kl_values_cylinders.append(kl_kde)

    # === Bar Plot of KL Divergences (subplot 8) ===
    ax_bar = axs[1, 3]
    x_labels = [str(d) for d in output_dims]
    x = np.arange(len(output_dims))
    bar_width = 0.35

    ax_bar.bar(x - bar_width/2, kl_values_lanes, width=bar_width,
               color=colors['lanes'], label=r"KL(Cones‖Lanes)")
    ax_bar.bar(x + bar_width/2, kl_values_cylinders, width=bar_width,
               color=colors['cylinders'], label=r"KL(Cones‖Cylinders)")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(x_labels)
    ax_bar.set_xlabel("Feature Dimension")
    ax_bar.set_ylabel("KL Divergence")
    ax_bar.set_title("KL Divergence Summary")
    ax_bar.grid(True)
    ax_bar.legend(fontsize=12)

    plt.suptitle("Activation KDEs and KL Divergence (Reference: Cones)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("activation_kde_KL_summary.png", dpi=300)
    plt.savefig("activation_kde_KL_summary.pdf", dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_and_kl()
