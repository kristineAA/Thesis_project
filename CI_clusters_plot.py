import numpy as np
import matplotlib.pyplot as plt


def plot_existing_clusters_for_day(
    daily,
    res,
    idx,
    alpha=0.25,
    B=20000,
    bins=30,
    scenario_names=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(0)

    row = daily.iloc[idx]
    m = int(row["m"])
    day = row["date"]

    # infer scenario names from dataframe if not given
    if scenario_names is None:
        center_cols = sorted([c for c in daily.columns if c.startswith("center_")])
        scenario_names = [c.replace("center_", "") for c in center_cols]

    k = len(scenario_names)

    # simulate again for visualization only
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2])
    central = sims[(sims >= lo) & (sims <= hi)]

    # use stored scenario centers relative to prediction
    centers = np.array(
        [row[f"center_{name}"] - row["predicted"] for name in scenario_names],
        dtype=float,
    )
    probs = np.array(
        [row[f"prob_{name}"] for name in scenario_names],
        dtype=float,
    )

    # sort by center value
    order = np.argsort(centers)
    centers = centers[order]
    probs = probs[order]
    scenario_names = [scenario_names[i] for i in order]

    # boundaries between clusters = midpoints between consecutive centers
    boundaries = [(centers[i] + centers[i + 1]) / 2 for i in range(k - 1)]

    # histogram of central simulated errors
    counts, edges = np.histogram(central, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(10, 6))

    cmap = plt.cm.get_cmap("viridis", k)
    colors = [cmap(i) for i in range(k)]

    # color each histogram bin by nearest cluster region
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
            label=f"{scenario_names[i]} (p={probs[i]:.2f})",
        )

    plt.axvline(lo, color="red", linestyle=":", linewidth=1.5, label="Lower CI bound")
    plt.axvline(hi, color="red", linestyle=":", linewidth=1.5, label="Upper CI bound")
    plt.axvline(0, color="black", linewidth=1, label="Zero error")

    plt.title(f"Daily Error Clusters | {day} | m={m}")
    plt.xlabel("Simulated Daily Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()