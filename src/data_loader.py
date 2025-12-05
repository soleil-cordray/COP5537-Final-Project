# ==============================================================================
# src/data_loader.py
#
# Everything will import this file
# ==============================================================================

# Treat type annotations as strings until later
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class BrainNetworkData:
    """
    Container for all core brain-network objects.
    """

    graph: nx.Graph  # Functional connectivity graph
    matrix: pd.DataFrame  # Full 90x90 correlation matrix
    region_labels: pd.DataFrame  # Region metadata (if available)
    system_by_region: Dict[str, str]  # Region -> brain system/network


# ==============================================================================
# BRAIN NETWORK LOADER HELPERS
# ==============================================================================


def _load_connectivity_matrix(path: Path) -> pd.DataFrame:
    """
    Load the 90×90 functional connectivity matrix from CSV & enforce symmetry.
    """
    if not path.exists():
        raise FileNotFoundError(f"Connectivity matrix not found at: {path}")

    df = pd.read_csv(path, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"Connectivity matrix must be square, got {df.shape[0]}x{df.shape[1]}"
        )

    # Enforce symmetry: correlations(i,j) and correlations(j,i) may differ
    # due to numerical noise; average each pair with its transpose entry
    df = (df + df.T) / 2.0
    return df


def _load_region_labels(path: Path) -> pd.DataFrame:
    """
    (A) IF AVAILABLE: Load region label metadata from CSV.
    (B) IF MISSING: Return an empty DataFrame.
    """
    if not path.exists():
        # Return a simple DataFrame later built from matrix index
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df


def _load_learning_networks(path: Path) -> Dict[str, str]:
    """
    Load region-to-system mappings from JSON.
    JSON FORMAT: Handles both system→regions and region→system.
    """
    if not path.exists():
        return {}

    with path.open("r") as f:
        raw = json.load(f)

    # Case 2: already region -> system
    if raw and isinstance(next(iter(raw.values())), str):
        return raw

    # Case 1: system -> [regions]; invert
    inv: Dict[str, str] = {}
    for system, regions in raw.items():
        for r in regions:
            inv[r] = system

    return inv


def _infer_region_labels_from_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple region labels from the matrix index.
    USE CASE: When no label file is available.
    """
    return pd.DataFrame(
        {
            "region_id": range(len(matrix.index)),
            "region_name": matrix.index.tolist(),
        }
    )


def build_graph_from_matrix(
    matrix: pd.DataFrame,
    weight_threshold: float = 0.2,
    use_absolute: bool = True,
) -> nx.Graph:
    """
    Convert the connectivity matrix into an undirected weighted graph.
    THRESHOLDING ENFORCED: Keeps only edges above a threshold.
    """
    G = nx.Graph()

    # Add all brain regions (nodes) using matrix indices (node labels).
    for region in matrix.index:
        G.add_node(region)

    values = matrix.values
    labels = list(matrix.index)

    # Decide which entries become edges:
    # - key = |w| if use_absolute else w
    # - We INCLUDE an edge if key >= weight_threshold,
    #   but we STORE the original signed weight w.
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(values[i, j])
            key = abs(w) if use_absolute else w
            if key >= weight_threshold:
                G.add_edge(labels[i], labels[j], weight=w)

    return G


# ==============================================================================
# BUILD CORE BRAIN NETWORK
# ==============================================================================


def load_brain_network(
    conn_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    systems_path: Optional[Path] = None,
    weight_threshold: float = 0.2,
    use_absolute: bool = True,
) -> BrainNetworkData:
    """
    (1) Load connectivity, labels, and system mappings.
    (2) Build a BrainNetworkData bundle with the graph.
    """
    conn_path = conn_path or (DATA_DIR / "functional_connectivity_90x90.csv")
    labels_path = labels_path or (DATA_DIR / "region_labels.csv")
    systems_path = systems_path or (DATA_DIR / "learning_networks.json")

    try:
        matrix = _load_connectivity_matrix(conn_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Connectivity matrix not found at: {conn_path}\n"
            f"Please run: python3 download_data.py"
        )

    region_labels = _load_region_labels(labels_path)
    system_by_region = _load_learning_networks(systems_path)

    # If no labels file, infer simple labels from matrix
    if region_labels.empty:
        region_labels = _infer_region_labels_from_matrix(matrix)

    # Normalize system_by_region keys to match matrix index (case-insensitive).
    normalized_system_by_region: Dict[str, str] = {}
    index_lower_to_label = {str(name).lower(): name for name in matrix.index}

    for region_key, system_name in system_by_region.items():
        region_key_lower = str(region_key).lower()
        if region_key_lower in index_lower_to_label:
            normalized_system_by_region[index_lower_to_label[region_key_lower]] = (
                system_name
            )
        else:
            # Keep as-is; might still be useful as a free name
            normalized_system_by_region[region_key] = system_name

    G = build_graph_from_matrix(
        matrix,
        weight_threshold=weight_threshold,
        use_absolute=use_absolute,
    )

    return BrainNetworkData(
        graph=G,
        matrix=matrix,
        region_labels=region_labels,
        system_by_region=normalized_system_by_region,
    )


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    brain = load_brain_network()
    print(
        f"Loaded graph with {brain.graph.number_of_nodes()} nodes "
        f"and {brain.graph.number_of_edges()} edges."
    )
