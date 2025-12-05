# ==============================================================================
# src/attention_dynamics.py
#
# Simulates how task-related attention spreads through brain networks over time:
# (1) Start from high-attention seed regions in the model.
# (2) Activate neighboring regions across strong connections
# (3) Record which areas of the brain become active at each discrete time step
#     - This measures "attention".
# (4) Generate a pattern timeline for learning-related systems.
# ==============================================================================

from __future__ import annotations

from typing import Dict, List, Set

import networkx as nx
import pandas as pd

from data_loader import RESULTS_DIR, BrainNetworkData


def simulate_spreading(
    G: nx.Graph,
    seed_nodes: List[str],
    steps: int = 6,
    weight_threshold: float = 0.25,
) -> Dict[int, List[str]]:
    """
    (1) Simulate spreading activation from seed regions over time.
        METHODOLOGY: Discrete time steps based on edge-weight thresholds.
    (2) Return which regions activate at each step.
    """
    active: Set[str] = set(seed_nodes)
    newly_active_by_step: Dict[int, List[str]] = {0: list(seed_nodes)}

    for t in range(1, steps + 1):
        new_active: Set[str] = set()
        for u in list(active):
            for v, data in G[u].items():
                if v in active:
                    continue
                w = abs(float(data.get("weight", 0.0)))
                if w >= weight_threshold:
                    new_active.add(v)
        if not new_active:
            break
        active.update(new_active)
        newly_active_by_step[t] = sorted(new_active)

    return newly_active_by_step


def run_attention_dynamics(brain: BrainNetworkData) -> None:
    """
    (1) Run an attention spreading simulation.
    (2) Save the activation timeline to CSV.
    """
    G = brain.graph

    # Choose seeds: e.g. all regions labelled "Visual" or similar
    seeds = [r for r, s in brain.system_by_region.items() if "Visual" in str(s)]
    # Fallback: just pick 3 highest-degree nodes if we have no visual label
    if not seeds:
        seeds = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:3]
        seeds = [s for s, _ in seeds]

    timeline = simulate_spreading(G, seeds, steps=8, weight_threshold=0.25)

    rows = []
    for t, nodes in timeline.items():
        for n in nodes:
            rows.append(
                {
                    "step": t,
                    "region": n,
                    "system": brain.system_by_region.get(n),
                }
            )

    df = pd.DataFrame(rows)

    out_dir = RESULTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attention_dynamics.csv"
    df.to_csv(out_path, index=False)

    print("Attention dynamics simulation complete.")
    print("  - Results →", out_path)
