import matplotlib.pyplot as plt
import numpy as np

# ============================
# Data
# ============================
stages = np.array([1, 2, 3, 4])

mAP = np.array([45.36, 42.19, 45.85, 54.81])
R1 = np.array([72.12, 67.48, 68.10, 75.26])


# ============================
# Nature-style configuration
# ============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0


# ============================
# Plot
# ============================
fig, ax = plt.subplots(figsize=(4.5, 3.2))


# Nature-like colors
color_map = "#0072B2"   # deep blue
color_r1 = "#D55E00"    # orange-red


ax.plot(
    stages,
    mAP,
    marker='o',
    markersize=6,
    linewidth=2.2,
    color=color_map,
    label='mAP'
)

ax.plot(
    stages,
    R1,
    marker='s',
    markersize=6,
    linewidth=2.2,
    color=color_r1,
    label='Rank-1'
)


# ============================
# Axis
# ============================
ax.set_xlabel("Learning Stage", fontsize=11)
ax.set_ylabel("Performance (%)", fontsize=11)

ax.set_xticks(stages)
ax.set_xticklabels(
    ['T1', 'T2', 'T3', 'T4']
)

ax.set_ylim(35, 85)

ax.legend(
    frameon=False,
    fontsize=10,
    loc='lower right'
)


# Remove unnecessary borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Add grid (very light)
ax.grid(
    linestyle='--',
    linewidth=0.5,
    alpha=0.3
)


plt.tight_layout()


# Save vector figure for paper
plt.savefig(
    "stagewise_performance.pdf",
    bbox_inches='tight'
)

plt.savefig(
    "stagewise_performance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()