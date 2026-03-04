import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"

data = pd.read_csv("stairs_data.csv")

# DaD + DeDoDev2,0,78,14.6,p

color_map = {
    # ===== Ours =====
    "SAK (Ours)": "#d62728",  # red
    "SAK (Ours Points only)": "#d62728",

    # ===== ALIKED family =====
    "ALIKED": "#2ca02c",  # green
    "ALIKED + DeepLSD": "#2ca02c",
    "ALIKED + MLSD":  "#17becf",   # cyan / turquoise
    "ALIKED + TPLSD": "#bcbd22",   # mustard / yellow-olive

    # ===== SuperPoint family =====
    "SuperPoint": "#1f77b4",  # blue
    "SuperPoint + DeepLSD": "#1f77b4",

    # ===== Wireframe =====
    "Wireframe": "#9467bd",  # purple

    # ===== PLNet =====
    "PLNet": "#8c564b",  # brown

    # ===== DISK =====
    "DISK": "#e377c2",  # pink

    # ===== DaD family =====
    "DaD + DeDoDev2": "#ff7f0e",  # orange
    "DaD + DeDoDev2 + ScaleLSD": "#ff7f0e",
}

# ---------------------------
# Figure with two subplots
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)


def plot_panel(ax, title, *, device):
    sak_coords = {}

    for _, row in data.iterrows():

        if "DeDoDev2" in row['Name'] and device == "CPU":
            continue

        name = row['Name']

        category = row['Category']
        color = color_map.get(name, 'gray')

        if name == "ALIKED + MLSD":
            marker, size = 's', 100
        elif name == "ALIKED + TPLSD":
            marker, size = 'p', 120
        elif 'SAK' in name and category == 'p+l':
            marker, size = '*', 400
        elif category == 'p':
            marker, size = 'o', 100
        else:
            marker, size = '^', 120

        print(row[f"Latency {device}"], row["Reconstruction Quality"])

        ax.scatter(
            float(row[f"Latency {device}"]),  # SAME values for CPU/GPU for now
            float(row["Reconstruction Quality"]),
            color=color,
            marker=marker,
            s=size,
            zorder=3
        )

        if 'SAK' in name:
            sak_coords[name] = (row[f"Latency {device}"], row["Reconstruction Quality"])

    # Connect SAK variants
    if 'SAK (Ours)' in sak_coords and 'SAK (Ours Points only)' in sak_coords:
        a = sak_coords['SAK (Ours)']
        b = sak_coords['SAK (Ours Points only)']
        ax.plot([a[0], b[0]], [a[1], b[1]],
                'r--', linewidth=1.5, alpha=0.7)

    ax.set_title(title, fontsize=13)
    # ax.set_xlim(0, 350)
    # ax.set_ylim(10, 60)
    ax.grid(True, linestyle='--', alpha=0.6)


# Plot CPU and GPU panels
plot_panel(axes[0], r"\textbf{CPU}", device="CPU")
plot_panel(axes[1], r"\textbf{GPU}", device="GPU")

# Axis labels
axes[0].set_xlabel(r"$\textbf{Latency (ms)}\boldsymbol{\downarrow}$", fontsize=13)
axes[1].set_xlabel(r"$\textbf{Latency (ms)}\boldsymbol{\downarrow}$", fontsize=13)
axes[0].set_ylabel(r"$\textbf{Accuracy (\%) @ 5cm/5}^\circ \boldsymbol{\uparrow}$", fontsize=13)

# ---------------------------
# Legend (shared)
# ---------------------------
HSPACE = r'$\hspace{0.5cm}$'

legend_elements = [
    # ===== Ours =====
    Line2D([0], [0], marker='*', linestyle='',
           markerfacecolor=color_map["SAK (Ours)"],
           markeredgecolor='w', markersize=15,
           label="SAK (Ours)"),

    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor=color_map["SAK (Ours Points only)"],
           markeredgecolor='w', markersize=8,
           label="SAK (Ours Points only)"),

    # ===== Wireframe / DISK =====
    Line2D([0], [0], marker='^', linestyle='',
           markerfacecolor=color_map["Wireframe"],
           markeredgecolor='w', markersize=8,
           label="Wireframe"),

    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor=color_map["DISK"],
           markeredgecolor='w', markersize=8,
           label="DISK"),

    # ===== SuperPoint family =====
    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor=color_map["SuperPoint"],
           markeredgecolor='w', markersize=8,
           label="SuperPoint"),

    Line2D([0], [0], marker='^', linestyle='',
           markerfacecolor=color_map["SuperPoint + DeepLSD"],
           markeredgecolor='w', markersize=8,
           label="SuperPoint + DeepLSD"),

    # ===== ALIKED family =====
    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor=color_map["ALIKED"],
           markeredgecolor='w', markersize=8,
           label="ALIKED"),

    Line2D([0], [0], marker='^', linestyle='',
           markerfacecolor=color_map["ALIKED + DeepLSD"],
           markeredgecolor='w', markersize=8,
           label="ALIKED + DeepLSD"),

    Line2D([0], [0], marker='s', linestyle='',
           markerfacecolor=color_map["ALIKED + MLSD"],
           markeredgecolor='w', markersize=8,
           label="ALIKED + MLSD"),

    Line2D([0], [0], marker='p', linestyle='',
           markerfacecolor=color_map["ALIKED + TPLSD"],
           markeredgecolor='w', markersize=8,
           label="ALIKED + TPLSD"),

    # ===== PLNet =====
    Line2D([0], [0], marker='^', linestyle='',
           markerfacecolor=color_map["PLNet"],
           markeredgecolor='w', markersize=8,
           label="PLNet"),

    # ===== DaD family =====
    #Line2D([0], [0], marker='o', linestyle='',
    #       markerfacecolor=color_map["DaD + DeDoDev2"],
    #       markeredgecolor='w', markersize=8,
    #       label="DaD + DeDoDev2"),

    Line2D([0], [0], marker='^', linestyle='',
           markerfacecolor=color_map["DaD + DeDoDev2 + ScaleLSD"],
           markeredgecolor='w', markersize=8,
           label="DaD + DeDoDev2 + ScaleLSD"),
]

leg = fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=11,
    frameon=True
)

# Render to place colored dots in legend
fig.canvas.draw()

plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("teaser_v3.pdf", format="pdf", bbox_inches="tight")
plt.show()
