# ==============================================================================
# src/hub_analysis.py
#
# Analyze highly connected nodes.
# ==============================================================================

from __future__ import annotations

import networkx as nx
import pandas as pd

from data_loader import RESULTS_DIR, BrainNetworkData


def compute_centralities(brain: BrainNetworkData) -> pd.DataFrame:
    """
    Compute degree, betweenness, closeness, and PageRank for each brain region.
    """
    G = brain.graph

    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, weight=None, normalized=True, endpoints=False)
    clo = nx.closeness_centrality(G)
    pr = nx.pagerank(G, weight="weight")

    rows = []
    for node in G.nodes():
        rows.append(
            {
                "region": node,
                "degree": deg.get(node, 0),
                "betweenness": bet.get(node, 0.0),
                "closeness": clo.get(node, 0.0),
                "pagerank": pr.get(node, 0.0),
                "system": brain.system_by_region.get(node, None),
            }
        )

    df = pd.DataFrame(rows)
    return df


def run_hub_analysis(brain: BrainNetworkData) -> None:
    """
    (1) Run hub analysis.
    (2) Save centrality metrics to CSV.
    (3) Report output location.
    """
    df = compute_centralities(brain)

    out_dir = RESULTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "learning_hubs.csv"
    df.to_csv(out_path, index=False)

    print("Hub analysis complete.")
    print("  - Results →", out_path)
