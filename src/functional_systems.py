# ==============================================================================
# src/functional_systems.py
#
# Detect brain network communities using modularity.
# ==============================================================================

from __future__ import annotations

from typing import AbstractSet, Sequence

import matplotlib.pyplot as plt
import networkx as nx

from data_loader import RESULTS_DIR, BrainNetworkData


def run_functional_systems(brain: BrainNetworkData) -> None:
    """
    (1) Detect communities in the brain network.
    (2) Save a simple community-labeled plot with modularity value.
    """
    G = brain.graph

    # Use built-in greedy modularity (no extra dependency)
    communities: Sequence[AbstractSet[str]] = (
        nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    )

    modularity = nx.algorithms.community.modularity(G, communities, weight="weight")

    # Save a simple community-labeled plot
    fig_path = RESULTS_DIR / "figures" / "functional_communities.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(G, seed=0, k=0.25)

    plt.figure(figsize=(8, 6))
    for idx, comm in enumerate(communities):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(comm),
            node_size=120,
            alpha=0.9,
            label=f"Community {idx + 1}",
        )
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    plt.axis("off")
    plt.title(f"Functional Communities (modularity Q ≈ {modularity:.2f})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Community detection complete.")
    print("  - Figure →", fig_path)
