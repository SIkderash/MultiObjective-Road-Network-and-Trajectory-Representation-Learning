import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Metric configuration grouped by task
tasks = {
    "Road Classification": [
        {
            "name": "Macro F1", "label": "Ma-F1", "data": ([0.68, 0.74, 0.78, 0.80], [0.66, 0.73, 0.77, 0.79]),
            "y_range": (0, 1), "is_higher_better": True
        },
        {
            "name": "Micro F1", "label": "Mi-F1", "data": ([0.72, 0.76, 0.81, 0.83], [0.71, 0.75, 0.80, 0.82]),
            "y_range": (0, 1), "is_higher_better": True
        },
    ],
    "Traffic Speed Inference": [
        {
            "name": "MAE (Speed)", "label": "MAE", "data": ([150, 140, 130, 125], [155, 145, 132, 128]),
            "y_range": (100, 160), "is_higher_better": False
        },
        {
            "name": "RMSE (Speed)", "label": "RMSE", "data": ([200, 190, 180, 175], [210, 195, 182, 178]),
            "y_range": (150, 220), "is_higher_better": False
        }
    ],
    "Travel Time Estimation": [
        {
            "name": "MAE (Time)", "label": "MAE", "data": ([60, 55, 49, 45], [62, 57, 50, 46]),
            "y_range": (40, 70), "is_higher_better": False
        },
        {
            "name": "RMSE (Time)", "label": "RMSE", "data": ([90, 84, 75, 70], [95, 86, 76, 72]),
            "y_range": (60, 100), "is_higher_better": False
        }
    ],
    "Trajectory Similarity": [
        {
            "name": "Hit Ratio @10", "label": "HR@10", "data": ([0.40, 0.48, 0.55, 0.60], [0.38, 0.47, 0.54, 0.58]),
            "y_range": (0, 1), "is_higher_better": True
        },
        {
            "name": "Mean Rank", "label": "MR", "data": ([80, 70, 60, 55], [82, 72, 62, 58]),
            "y_range": (40, 90), "is_higher_better": False
        }
    ]
}

# Model variants
components = [
    "Base (MTM)",
    "MTM + Spatial Fusion",
    "MTM + Spatial Fusion + Contrastive",
    "MTM + Spatial Fusion + Contrastive + Adaptive NegSampling"
]

# Colors
base_rgb = np.array(mcolors.to_rgb('tab:blue'))
white = np.array([1.0, 1.0, 1.0])
alphas = np.linspace(0.25, 1.0, len(components))
shades = [(1 - alpha) * white + alpha * base_rgb for alpha in alphas]

# Plot parameters
n = 1.5
bar_width = 0.15
group_gap = 0.1
label_offset = 0.015

# Create separate figures per task
for task_name, metrics in tasks.items():
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    x_group_centers = [0, len(components) * bar_width + group_gap]

    for ax, metric in zip(axs, metrics):
        dataset1_values, dataset2_values = metric["data"]
        for i, color in enumerate(shades):
            x1 = x_group_centers[0] + i * bar_width
            x2 = x_group_centers[1] + i * bar_width
            val1 = dataset1_values[i]
            val2 = dataset2_values[i]
            ax.bar(x1, val1, width=bar_width, color=color)
            ax.bar(x2, val2, width=bar_width, color=color)

            offset = (metric["y_range"][1] - metric["y_range"][0]) * label_offset
            ax.text(x1, val1 + offset, f"{val1:.2f}", ha='center', va='bottom', fontsize=6 * n)
            ax.text(x2, val2 + offset, f"{val2:.2f}", ha='center', va='bottom', fontsize=6 * n)

        ax.set_xticks([
            x_group_centers[0] + (len(components) * bar_width) / 2,
            x_group_centers[1] + (len(components) * bar_width) / 2
        ])
        ax.set_xticklabels(["Chengdu", "Xian"], fontsize=10 * n)
        # ax.set_title(metric["name"], fontsize=12 * n)
        ax.set_ylabel(metric["label"], fontsize=10 * n, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10 * n)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(*metric["y_range"])

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in shades]

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    filename = f"Ablation Images/{task_name.lower().replace(' ', '_')}_ablation.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.colors as mcolors

# Model variants
components = [
    "Base (MTM)",
    "MTM + Spatial Fusion",
    "MTM + Spatial Fusion + Contrastive",
    "MTM + Spatial Fusion + Contrastive + Adaptive NegSampling"
]

# Generate light-to-dark blue shades
base_rgb = np.array(mcolors.to_rgb('tab:blue'))
white = np.array([1.0, 1.0, 1.0])
alphas = np.linspace(0.25, 1.0, len(components))
shades = [(1 - alpha) * white + alpha * base_rgb for alpha in alphas]

# Create custom legend handles
handles = [mpatches.Patch(color=color, label=label) for label, color in zip(components, shades)]

# Create a dummy figure just for the legend
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')  # Hide axes

# Add the legend
legend = fig.legend(
    handles=handles,
    title="Model Variants",
    loc='center',
    fontsize=12,
    title_fontsize=13,
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig("Ablation Images/legend_only.png", dpi=300, bbox_inches='tight')

