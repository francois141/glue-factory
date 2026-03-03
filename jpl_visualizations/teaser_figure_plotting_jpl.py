import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"

data = pd.read_csv("stairs_data.csv")

color_map = {
    'SAK (Ours)': 'red',
    'SAK (Ours, Points only)': 'red',
    'SuperPoint + DeepLSD': 'blue',
    'SuperPoint': 'blue',
    'ALIKED + DeepLSD': 'green',
    'ALIKED': 'green',
    'Wireframe': 'purple',
    'DISK': 'purple',
    'DaD + DeDoDev2 + ScaleLSD': 'orange',
    'DaD + DeDoDev2': 'orange',
    'ALIKED + ScaleLSD': 'brown',
    'SuperPoint + ScaleLSD': 'darkblue'
}

# ---------------------------
# Figure with two subplots
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

def plot_panel(ax, title):
    sak_coords = {}

    for _, row in data.iterrows():
        name = row['Name']
        category = row['Category']
        color = color_map.get(name, 'gray')

        if 'SAK' in name and category == 'p+l':
            marker, size = '*', 400
        elif category == 'p':
            marker, size = 'o', 100
        else:
            marker, size = '^', 120

        print(row["Latency CPU"], row["Reconstruction Quality"])

        ax.scatter(
            row["Latency CPU"],               # SAME values for CPU/GPU for now
            row["Reconstruction Quality"],
            color=color,
            marker=marker,
            s=size,
            zorder=3
        )

        if 'SAK' in name:
            sak_coords[name] = (row["Latency CPU"], row["Reconstruction Quality"])

    # Connect SAK variants
    if 'SAK (Ours)' in sak_coords and 'SAK (Ours, Points only)' in sak_coords:
        a = sak_coords['SAK (Ours)']
        b = sak_coords['SAK (Ours, Points only)']
        ax.plot([a[0], b[0]], [a[1], b[1]],
                'r--', linewidth=1.5, alpha=0.7)

    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 350)
    ax.set_ylim(10, 60)
    ax.grid(True, linestyle='--', alpha=0.6)


# Plot CPU and GPU panels
plot_panel(axes[0], r"\textbf{CPU}")
plot_panel(axes[1], r"\textbf{GPU}")

# Axis labels
axes[0].set_xlabel(r"$\textbf{Latency (ms)}\boldsymbol{\downarrow}$", fontsize=13)
axes[1].set_xlabel(r"$\textbf{Latency (ms)}\boldsymbol{\downarrow}$", fontsize=13)
axes[0].set_ylabel(r"$\textbf{Accuracy (\%) @ 5cm/5}^\circ \boldsymbol{\uparrow}$", fontsize=13)

# ---------------------------
# Legend (shared)
# ---------------------------
HSPACE = r'$\hspace{0.5cm}$'
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='SAK (Ours)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='SAK (Ours, Points only)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=8, label='Wireframe'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='DISK'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label=f'SuperPoint'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=8, label=f'SuperPoint + DeepLSD'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='darkblue', markersize=8, label='SuperPoint + ScaleLSD'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label=f'ALIKED'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label=f'ALIKED + DeepLSD'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='brown', markersize=8, label='ALIKED + ScaleLSD'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label=f'DaD + DeDoDev2'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label=f'DaD + DeDoDev2 + ScaleLSD'),
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