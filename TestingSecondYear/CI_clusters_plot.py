import numpy as np
import matplotlib.pyplot as plt
from CI_clusters import *

def plot_existing_clusters_for_day(
    daily,
    res,
    idx,
    alpha=0.10,
    B=20000,
    bins=30,
    scenario_names=None,
):
    row = daily.iloc[idx]
    m = int(row["m"])
    day = row["date"]

    # default names if none provided
    if scenario_names is None:
        center_cols = sorted([c for c in daily.columns if c.startswith("center_")])
        scenario_names = [c.replace("center_", "") for c in center_cols]

    k = len(scenario_names)

    # simulate again for visualization only
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2])
    central = sims[(sims >= lo) & (sims <= hi)]

    # fetch centers/probabilities dynamically
    centers = np.array([row[f"err_center_{name}"] for name in scenario_names], dtype=float)
    probs = np.array([row[f"prob_{name}"] for name in scenario_names], dtype=float)

    # sort just to be safe
    order = np.argsort(centers)
    centers = np.array([row[f"center_{name}"] - row["predicted"] for name in scenario_names], dtype=float)
    #centers = centers[order]
    probs = probs[order]
    scenario_names = [scenario_names[i] for i in order]

    # boundaries between clusters = midpoints between consecutive centers
    boundaries = [(centers[i] + centers[i + 1]) / 2 for i in range(k - 1)]

    # histogram
    counts, edges = np.histogram(central, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(10, 6))

    # automatic colors from matplotlib colormap
    cmap = plt.cm.get_cmap("viridis", k)
    colors = [cmap(i) for i in range(k)]

    # color each bin according to which cluster region it belongs to
    for i in range(len(counts)):
        x = bin_centers[i]

        cluster_idx = 0
        while cluster_idx < len(boundaries) and x > boundaries[cluster_idx]:
            cluster_idx += 1

        plt.bar(
            edges[i],
            counts[i],
            width=edges[i + 1] - edges[i],
            align="edge",
            color=colors[cluster_idx],
            alpha=0.6,
        )

    # plot cluster centers
    for i in range(k):
        plt.axvline(
            centers[i],
            linestyle="--",
            color=colors[i],
            linewidth=2,
            label=f"{scenario_names[i]} (p={probs[i]:.2f})"
        )

    plt.axvline(0, color="black", linewidth=1)

    plt.title(f"Daily Error Clusters | {day} | m={m}")
    plt.xlabel("Simulated Daily Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

