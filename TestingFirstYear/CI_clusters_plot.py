import numpy as np
import matplotlib.pyplot as plt
from CI_clusters import *

def plot_existing_clusters_for_day(daily, res, idx, alpha=0.10, B=20000, bins=30):
    
    row = daily.iloc[idx]
    m = int(row["m"])
    day = row["date"]

    # simulate (visualization only)
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)
    lo, hi = np.quantile(sims, [alpha/2, 1-alpha/2])
    central = sims[(sims >= lo) & (sims <= hi)]

    centers = np.array(row["centers_low_mid_high"])
    probs = [row["p_low"], row["p_mid"], row["p_high"]]

    # --- compute cluster boundaries (midpoints between centers) ---
    boundaries = [
        (centers[0] + centers[1]) / 2,
        (centers[1] + centers[2]) / 2
    ]

    # histogram values
    counts, edges = np.histogram(central, bins=bins)

    # bin centers
    bin_centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(8,5))

    colors = ["red","gray","green"]
    names = ["Low","Mid","High"]

    # color each bin according to region
    for i in range(len(counts)):
        x = bin_centers[i]

        if x <= boundaries[0]:
            color = colors[0]
        elif x <= boundaries[1]:
            color = colors[1]
        else:
            color = colors[2]

        plt.bar(
            edges[i],
            counts[i],
            width=edges[i+1] - edges[i],
            align='edge',
            color=color,
            alpha=0.6
        )

    # plot cluster centers
    for i in range(3):
        plt.axvline(
            centers[i],
            linestyle="--",
            color=colors[i],
            linewidth=2,
            label=f"{names[i]} (p={probs[i]:.2f})"
        )

    plt.axvline(0, color="black", linewidth=1)

    plt.title(f"Daily Error Clusters | {day} | m={m}")
    plt.xlabel("Simulated Daily Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

