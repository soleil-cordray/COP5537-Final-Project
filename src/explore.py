# ==============================================================================
# src/explore.py
#
# Produces:
# - Network stats JSON (results/data/network_statistics.json)
# - Connectivity heatmap figure
# - Simple degree-distribution plot
# - Quick network visualization with hubs highlighted
# ==============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from data_loader import (
    RESULTS_DIR,
    BrainNetworkData,
    load_brain_network,
)

# ==============================================================================
# COMPUTATION
# ==============================================================================


def compute_basic_stats(brain: BrainNetworkData) -> Dict[str, Any]:
    """
    Compute core network statistics.
    STATS: node/edge counts, density, clustering,
           whether connected/disconnected, largest component length/size.
    """
    G = brain.graph

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)

    # Largest connected component
    if nx.is_connected(G):
        GCC = G
    else:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        GCC = G.subgraph(components[0]).copy()

    # Average shortest path over GCC only
    try:
        avg_path_length = nx.average_shortest_path_length(GCC, weight=None)
    except nx.NetworkXError:
        avg_path_length = float("nan")

    avg_clustering = nx.average_clustering(G, weight="weight")

    stats = {
        "n_nodes": n,
        "n_edges": m,
        "density": float(density),
        "avg_clustering": float(avg_clustering),
        "avg_path_length_largest_component": float(avg_path_length),
        "is_connected": nx.is_connected(G),
        "largest_component_size": GCC.number_of_nodes(),
    }

    return stats


def save_stats(stats: Dict[str, Any], out_path: Path) -> None:
    """
    Save computed network statistics as a JSON file at the given path.
    PURPOSE: Preserve results of analysis for future reference/processing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)


# ==============================================================================
# PLOTTING# ==============================================================================


def plot_connectivity_heatmap(brain: BrainNetworkData, out_path: Path) -> None:
    """
    Save heatmap of functional connectivity matrix to given image path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        brain.matrix,
        cmap="RdBu_r",
        center=0.0,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Functional Connectivity (90×90)")
    plt.xlabel("Region Index")
    plt.ylabel("Region Index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_degree_distribution(brain: BrainNetworkData, out_path: Path) -> None:
    """
    Plot & save histogram showing degree distribution of brain graph.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    G = brain.graph
    degrees = [deg for _, deg in G.degree()]

    plt.figure(figsize=(6, 4))
    plt.hist(degrees, bins=15)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Degree Distribution of Brain Network")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_network_overview(brain: BrainNetworkData, out_path: Path) -> None:
    """
    Draw & save spring-layout overview of brain network.
    Highlights hubs & systems.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    G = brain.graph

    # Node sizes proportional to degree
    degrees = dict(G.degree())  # maps nodes to their degrees
    max_deg = max(degrees.values()) if degrees else 1
    sizes = [300 + 700 * (deg / max_deg) for deg in degrees.values()]

    # Color nodes by system if available, else single color
    system_by_region = brain.system_by_region
    systems = sorted(set(system_by_region.values())) if system_by_region else []
    system_to_color = {sys: idx for idx, sys in enumerate(systems)}

    node_colors = []
    for node in G.nodes():
        sys = system_by_region.get(node)
        if sys is None:
            node_colors.append(0.5)
        else:
            node_colors.append(system_to_color.get(sys, 0.5))

    pos = nx.spring_layout(G, seed=42, k=0.25)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=node_colors,
        cmap="viridis",
        alpha=0.9,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    plt.axis("off")
    plt.title("Brain Network Overview (size ≈ degree)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==============================================================================
# ANALYSIS ==============================================================================


def run_exploratory_analysis(brain: BrainNetworkData) -> None:
    """
    (1) Run exploratory analysis
    (2) Generate summary statistics & figures for brain network.
    """
    data_dir = RESULTS_DIR / "data"
    fig_dir = RESULTS_DIR / "figures"

    stats = compute_basic_stats(brain)
    save_stats(stats, data_dir / "network_statistics.json")

    plot_connectivity_heatmap(brain, fig_dir / "connectivity_matrix.png")
    plot_degree_distribution(brain, fig_dir / "degree_distribution.png")
    plot_network_overview(brain, fig_dir / "brain_network_learning.png")

    print("Exploratory analysis complete.")
    print("  - Stats  →", data_dir / "network_statistics.json")
    print("  - Figures →", fig_dir)


if __name__ == "__main__":
    brain = load_brain_network()
    run_exploratory_analysis(brain)
