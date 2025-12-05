# ==============================================================================
# src/main.py
#
# Complete analysis pipeline for this project's brain network study.
# ==============================================================================

from __future__ import annotations

from attention_dynamics import run_attention_dynamics
from data_loader import load_brain_network
from explore import run_exploratory_analysis
from functional_systems import run_functional_systems
from hub_analysis import run_hub_analysis
from learning_pathways import run_learning_pathways
from mental_health_sim import run_mental_health_sim


def main() -> None:
    """
    (1) Run the full brain network analysis pipeline.
    (2) Print where outputs are stored.
    """
    print("=== Brain Networks for Learning: Analysis Pipeline ===")
    brain = load_brain_network()
    print(
        f"Loaded graph with {brain.graph.number_of_nodes()} nodes and "
        f"{brain.graph.number_of_edges()} edges."
    )

    print("\n[1/6] Exploratory analysis...")
    run_exploratory_analysis(brain)

    print("\n[2/6] Learning pathways (max-flow)...")
    run_learning_pathways(brain)

    print("\n[3/6] Hub analysis (centralities)...")
    run_hub_analysis(brain)

    print("\n[4/6] Mental health simulations...")
    run_mental_health_sim(brain)

    print("\n[5/6] Attention dynamics (spreading)...")
    run_attention_dynamics(brain)

    print("\n[6/6] Functional systems (communities)...")
    run_functional_systems(brain)

    print("\nPipeline complete. Check the results/ directory for figures and data.")


if __name__ == "__main__":
    main()
