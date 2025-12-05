# ==============================================================================
# src/mental_health_sim.py
#
# Analyze the impact of depression, anxiety, and ADHD on brain networks. ================================================================================

from __future__ import annotations

from copy import deepcopy
from typing import Dict

import networkx as nx
import pandas as pd

from data_loader import RESULTS_DIR, BrainNetworkData
from hub_analysis import compute_centralities

# ==============================================================================
# SCALE FACTORS
# Based on neuroscience literature:
# (A) Depression:
#     - DMN hyperconnectivity: 1.3x (Kaiser et al., 2015)
#     - Executive control weakness: 0.8x
#     - Reduced anti-correlation: 1.1x (less suppression)
# (B) Anxiety:
#     - Salience hyperactivity: 1.3x (Paulus & Stein, 2006)
#     - Attention disruption: 1.2x
# (C) ADHD:
#     - Executive/Attention deficits: 0.7x (Castellanos et al., 2008)
#     - Reduced DMN suppression: 1.1x
# ==============================================================================


def _scale_edges_between(
    G: nx.Graph,
    system_by_region: Dict[str, str],
    system_a: str,
    system_b: str,
    factor: float,
) -> None:
    """
    Multiply edge weights connecting two given systems by the specified factor.
    """
    for u, v, data in G.edges(data=True):
        sa = system_by_region.get(u)
        sb = system_by_region.get(v)
        if {sa, sb} == {system_a, system_b}:
            data["weight"] *= factor


def _scale_edges_within(
    G: nx.Graph,
    system_by_region: Dict[str, str],
    system: str,
    factor: float,
) -> None:
    """
    Multiply edges weights of same-system endpoints by the given factor.
    """
    for u, v, data in G.edges(data=True):
        if system_by_region.get(u) == system and system_by_region.get(v) == system:
            data["weight"] *= factor


# ==============================================================================
# SIMULATION
# ==============================================================================


def simulate_condition(brain: BrainNetworkData, condition: str) -> pd.DataFrame:
    """
    (1) Apply condition-specific weight changes to the graph.
        - Depression: STRONGER DMN, WEAKER Executive/Attention
        - Anxiety: STRONGER Salience, WEAKER Executive paths
        - ADHD: WEAKER Executive + Attention + DMN suppression
    (2) Return centralities for the perturbed network.
        - e.g., stronger DMN for depression.
    """
    G = deepcopy(brain.graph)
    systems = brain.system_by_region

    cond = condition.lower()

    if "depress" in cond:
        _scale_edges_within(G, systems, "DefaultMode", 1.3)
        _scale_edges_between(G, systems, "DefaultMode", "ExecutiveControl", 1.1)
        _scale_edges_within(G, systems, "ExecutiveControl", 0.8)
        _scale_edges_within(G, systems, "Attention", 0.85)

    elif "anx" in cond:
        _scale_edges_within(G, systems, "Salience", 1.3)
        _scale_edges_between(G, systems, "Salience", "Attention", 1.2)
        _scale_edges_within(G, systems, "ExecutiveControl", 0.85)

    elif "adhd" in cond:
        _scale_edges_within(G, systems, "ExecutiveControl", 0.7)
        _scale_edges_within(G, systems, "Attention", 0.7)
        _scale_edges_between(G, systems, "DefaultMode", "ExecutiveControl", 1.1)

    # Compute centralities on perturbed (modified to condition) graph
    perturbed_brain = BrainNetworkData(
        graph=G,
        matrix=brain.matrix,
        region_labels=brain.region_labels,
        system_by_region=brain.system_by_region,
    )
    df = compute_centralities(perturbed_brain)
    df["condition"] = condition
    return df


def run_mental_health_sim(brain: BrainNetworkData) -> None:
    """
    (1) Simulate several mental health conditions.
    (2) Save centrality and summary metrics across conditions.
    """
    if not brain.system_by_region:
        print("Skipping mental health sim: no system labels found.")
        return

    dfs = []
    for cond in ["Healthy", "Depression", "Anxiety", "ADHD"]:
        if cond == "Healthy":
            df = compute_centralities(brain)
            df["condition"] = cond
        else:
            df = simulate_condition(brain, cond)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    # Example "learning efficiency" metric: mean PageRank of learning systems /
    # mean PageRank of DMN, per condition.
    def is_learning(sys: str) -> bool:
        return sys in {"ExecutiveControl", "Attention", "Memory"}

    summaries = []
    for cond, group in full.groupby("condition"):
        learning_mask = group["system"].apply(lambda s: is_learning(str(s)))
        dmn_mask = group["system"].eq("DefaultMode")

        pr_learning = group.loc[learning_mask, "pagerank"].mean()
        pr_dmn = group.loc[dmn_mask, "pagerank"].mean()
        ratio = float(pr_learning / pr_dmn) if pr_dmn > 0 else float("nan")

        summaries.append(
            {
                "condition": cond,
                "mean_pagerank_learning": float(pr_learning),
                "mean_pagerank_dmn": float(pr_dmn),
                "learning_efficiency_ratio": ratio,
            }
        )

    out_dir = RESULTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    full.to_csv(out_dir / "centralities_by_condition.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "mental_health_summary.csv", index=False)

    print("Mental health simulations complete.")
    print("  - Node-level centralities →", out_dir / "centralities_by_condition.csv")
    print("  - Summary metrics         →", out_dir / "mental_health_summary.csv")
