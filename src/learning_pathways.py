# src/learning_pathways.py

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import networkx as nx

from data_loader import RESULTS_DIR, BrainNetworkData


def _super_source_sink_flow(
    G: nx.Graph,
    sources: List[str],
    targets: List[str],
    capacity_attr: str = "weight",
) -> Tuple[float, Dict]:
    """
    Compute max-flow between two groups of nodes using an auxiliary graph.
    (1) Add a super-source node (one fake node connected TO all source nodes)
            & a super-sink node (one fake node connected FROM all target nodes)
    (2) Run max-flow from super-source group -> super-sink group
    (3) RESULT: total maximum information capacity from the source -> target
    """
    # Convert to directed
    DG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        w = float(data.get(capacity_attr, 1.0))
        if w <= 0:
            continue
        DG.add_edge(u, v, capacity=w)
        DG.add_edge(v, u, capacity=w)

    super_s = "__SOURCE__"
    super_t = "__SINK__"

    for s in sources:
        if s in DG:
            DG.add_edge(super_s, s, capacity=float("inf"))

    for t in targets:
        if t in DG:
            DG.add_edge(t, super_t, capacity=float("inf"))

    flow_value, flow_dict = nx.maximum_flow(DG, super_s, super_t)
    return flow_value, flow_dict


def run_learning_pathways(brain: BrainNetworkData) -> None:
    """
    Placeholder implementation:
    - picks example systems if available
    - computes max-flow between them
    - writes simple JSON with flow values
    """
    G = brain.graph
    system_by_region = brain.system_by_region

    # Fallback: nothing to do if no system labels
    if not system_by_region:
        print("Skipping learning_pathways: no system_by_region info.")
        return

    # Basic mapping: adjust these names to match your JSON if needed
    # e.g. "Attention", "Memory", "ExecutiveControl", "DefaultMode"
    attention_regions = [r for r, s in system_by_region.items() if "Attention" in s]
    memory_regions = [r for r, s in system_by_region.items() if "Memory" in s]

    if not attention_regions or not memory_regions:
        print("Skipping learning_pathways: could not find Attention/Memory systems.")
        return

    flow_value, _ = _super_source_sink_flow(G, attention_regions, memory_regions)

    out_dir = RESULTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "learning_pathways.json"

    payload = {
        "attention_regions": attention_regions,
        "memory_regions": memory_regions,
        "attention_to_memory_max_flow": float(flow_value),
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print("Learning pathway analysis complete.")
    print("  - Results →", out_path)
