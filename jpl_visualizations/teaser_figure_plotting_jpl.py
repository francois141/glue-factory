import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle



plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
data = pd.read_csv("stairs_data.csv")

# Define color mapping for method pairs
# Each p+l method and its corresponding p method get the same color
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
    'ALIKED + ScaleLSD': 'darkgreen',
    'SuperPoint + ScaleLSD': 'darkblue'
}

# Create the scatter plot
plt.figure(figsize=(6, 5))

# Store SAK coordinates for connecting line
sak_coords = {}

for _, row in data.iterrows():
    #if "DaD" in row['Name']:
    #    continue
    name = row['Name']
    category = row['Category']
    color = color_map.get(name, 'gray')

    # Determine marker and size
    if 'SAK' in name and category == 'p+l':
        # SAK (ours) gets a bigger star
        marker = '*'
        size = 400
    elif category == 'p':
        # Points only methods get circles
        marker = 'o'
        size = 100
    else:  # category == 'p+l'
        # Points + lines methods get triangles
        marker = '^'
        size = 120

    plt.scatter(row["Latency"], row["Reconstruction Quality"],
                label=name, color=color, marker=marker, s=size)

    # Store SAK coordinates
    if 'SAK' in name:
        sak_coords[name] = (row["Latency"], row["Reconstruction Quality"])

# Draw dotted line connecting the two SAK configs
if 'SAK (Ours)' in sak_coords and 'SAK (Ours, Points only)' in sak_coords:
    sak_ours = sak_coords['SAK (Ours)']
    sak_points = sak_coords['SAK (Ours, Points only)']
    plt.plot([sak_ours[0], sak_points[0]], [sak_ours[1], sak_points[1]],
             'r--', linewidth=1.5, alpha=0.7)

# Add vertical line at 100ms for 10FPS
plt.axvline(x=100, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
plt.text(124, plt.ylim()[0] + 30, r'$\textbf{10 fps}$', ha='center', va='bottom', fontsize=13)

# Add axis labels and title
plt.xlabel(r"$\textbf{Latency (ms) }\boldsymbol{\downarrow}$", fontsize=13)
plt.ylabel(r"$\textbf{Accuracy (\%) @ 5cm/5}^\circ \boldsymbol{\uparrow}$", fontsize=13)
#plt.title("Latency vs Reconstruction Quality")

plt.xlim(0, 350)   # e.g. from 0 to 400 ms
plt.ylim(10, 60)   # e.g. from 0% to 60%

# Create custom legend with combined markers
# Use hspace placeholder in labels; after rendering, overlay colored circle markers
HSPACE = r'$\hspace{0.5cm}$'
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='SAK (Ours)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='SAK (Ours, Points only)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=8, label='Wireframe'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='DISK'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=8, label=f'SuperPoint ({HSPACE}) + DeepLSD'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='darkblue', markersize=8, label='SuperPoint + ScaleLSD'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label=f'ALIKED ({HSPACE}) + DeepLSD'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='darkgreen', markersize=8, label='ALIKED + ScaleLSD'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label=f'DaD + DeDoDev2 ({HSPACE}) + ScaleLSD'),
]

leg = plt.legend(handles=legend_elements, loc="lower right", fontsize=10)

# Draw colored circle markers on top of the hspace gaps in the legend
fig = plt.gcf()
fig.canvas.draw()  # render so we can get text positions

# Map: legend entry index -> (color, x_offset, y_offset) in display pixels
# Adjust x_offset and y_offset manually to position each dot in the hspace
dot_entries = {
    4: ('blue',   235, 10),   # SuperPoint
    6: ('green',  206, 10),   # ALIKED
    8: ('orange', 325, 10),   # DaD + DeDoDev2
}

renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
for idx, (color, x_off, y_off) in dot_entries.items():
    text_obj = leg.get_texts()[idx]
    bb = text_obj.get_window_extent(renderer)
    dot_x = bb.x0 + x_off
    dot_y = (bb.y0 + bb.y1) / 2 + y_off
    fx, fy = inv.transform((dot_x, dot_y))
    fig.patches.append(Circle((fx, fy), 0.009, transform=fig.transFigure,
                              fc=color, ec=color, zorder=10))

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()