import matplotlib.pyplot as plt
import numpy as np

# Image resolutions
sizes = np.array([64, 128, 256, 512, 1024, 2048])

# Benchmark results (replace with your real data)
results = {
    "SuperPoint + DeepLSD": [293, 293, 293, 293, 293, 293],
    "ALIKED + DeepLSD": [286, 286, 286, 286, 286, 286],
    "DaD + DeDoDe v2 + ScaleLSD": [147, 147, 147, 147, 147, 147],
    "Wireframe": [266, 266, 266, 266, 266, 266],
    "PLNet": [250]*6,
    "SAK (Ours)": [120, 125, 130, 140, 150, 160],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
})

fig, ax = plt.subplots(figsize=(4, 2.7))

markers = ["o", "s", "D", "^", "v", "*"]
linestyles = ["-", "--", "-.", ":", "-", "-"]

for (model, values), m, ls in zip(results.items(), markers, linestyles):
    ax.plot(sizes, values, marker=m, linestyle=ls, label=model)

# Axes labels
ax.set_xlabel("Input resolution (pixels)")
ax.set_ylabel("Inference time (ms)")

# Log scale for X-axis
ax.set_xscale("log", base=2)
ax.set_xticks(sizes)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Set Y-axis limits
ax.set_ylim(0, 500)

# Grid
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

# Legend
ax.legend(frameon=False, loc="upper left")

plt.tight_layout(pad=0.3)

# Save figure
plt.savefig("benchmark_models.pdf", bbox_inches="tight")
plt.savefig("benchmark_models.svg", bbox_inches="tight")

plt.show()