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
    # Create a reproducible random number generator if none is provided
    if rng is None:
        rng = np.random.default_rng(0)

    # Select the row corresponding to the chosen day
    row = daily.iloc[idx]
    m = int(row["m"])
    day = row["date"]

    # If scenario names are not provided, infer them from columns named center_*
    if scenario_names is None:
        center_cols = sorted([c for c in daily.columns if c.startswith("center_")])
        scenario_names = [c.replace("center_", "") for c in center_cols]

    # Number of scenarios/clusters
    k = len(scenario_names)

    # Simulate daily errors by sampling m residuals with replacement
    # This is only used for visualization, not for recalculating the scenarios
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)

    # Keep only the central part of the simulated error distribution
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2])
    central = sims[(sims >= lo) & (sims <= hi)]

    # Convert stored scenario centers into errors relative to the daily prediction
    centers = np.array(
        [row[f"center_{name}"] - row["predicted"] for name in scenario_names],
        dtype=float,
    )

    # Extract the probability assigned to each scenario
    probs = np.array(
        [row[f"prob_{name}"] for name in scenario_names],
        dtype=float,
    )

    # Sort scenarios from lowest to highest simulated error
    order = np.argsort(centers)
    centers = centers[order]
    probs = probs[order]
    scenario_names = [scenario_names[i] for i in order]

    # Define cluster boundaries as midpoints between neighboring scenario centers
    boundaries = [(centers[i] + centers[i + 1]) / 2 for i in range(k - 1)]

    # Compute histogram values for the central simulated errors
    counts, edges = np.histogram(central, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Use one color per cluster
    cmap = plt.cm.get_cmap("viridis", k)
    colors = [cmap(i) for i in range(k)]

    # Plot each histogram bin and color it according to its nearest cluster region
    for i in range(len(counts)):
        x = bin_centers[i]

        # Find which cluster region the bin belongs to
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

    # Plot vertical lines for the scenario centers
    for i in range(k):
        plt.axvline(
            centers[i],
            linestyle="--",
            color=colors[i],
            linewidth=2,
            label=f"{scenario_names[i]} (p={probs[i]:.2f})",
        )

    # Plot interval bounds and the zero-error reference line
    plt.axvline(lo, color="red", linestyle=":", linewidth=1.5, label="Lower CI bound")
    plt.axvline(hi, color="red", linestyle=":", linewidth=1.5, label="Upper CI bound")
    plt.axvline(0, color="black", linewidth=1, label="Zero error")

    # Add title, axis labels, legend, and layout adjustment
    plt.title(f"Daily Error Clusters | {day} | m={m}")
    plt.xlabel("Simulated Daily Error")
    plt.ylabel("Frequency")
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()