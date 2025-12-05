# ==============================================================================
# download_data.py
#
# Download brain connectivity data for network analysis.
# ==============================================================================

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from nilearn import datasets
    from nilearn.connectome import ConnectivityMeasure

    try:
        from nilearn.maskers import NiftiLabelsMasker
    except ImportError:
        from nilearn.input_data import NiftiLabelsMasker
except ImportError:
    print("ERROR: nilearn not installed. Run: pip install nilearn")
    exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ==============================================================================
# DATA DOWNLOAD
# ==============================================================================


def download_fmri_data():
    """
    Download resting-state fMRI data.
    Uses publicly available dataset with HCP-compatible processing.
    """
    print("=" * 70)
    print("DOWNLOADING BRAIN CONNECTIVITY DATA")
    print("=" * 70)
    print("\nDataset: Development fMRI (young adult subset)")
    print("Processing: HCP-compatible preprocessing")
    print("Atlas: AAL (90 regions)\n")

    print("Step 1/5: Downloading fMRI data (2-5 min)...")
    data = datasets.fetch_development_fmri(
        n_subjects=1, reduce_confounds=True, age_group="adult"
    )
    print("✓ Downloaded\n")

    return data


# ==============================================================================
# CONNECTIVITY COMPUTATION
# ==============================================================================


def compute_connectivity(data):
    """
    Compute functional connectivity matrix using AAL atlas.
    Returns correlation matrix and region labels.
    """
    print("Step 2/5: Applying AAL atlas...")

    # Handle SSL certificate issues (common on Mac)
    try:
        atlas = datasets.fetch_atlas_aal()
    except Exception as ssl_error:
        if "SSL" in str(ssl_error) or "certificate" in str(ssl_error).lower():
            print("  SSL certificate error detected. Applying fix...")
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            atlas = datasets.fetch_atlas_aal()
        else:
            raise

    masker = NiftiLabelsMasker(
        labels_img=atlas["maps"],
        standardize=True,
        verbose=0,
    )

    print("Step 3/5: Extracting time series...")
    time_series = masker.fit_transform(data.func[0])

    print("Step 4/5: Computing correlation matrix...")
    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    # Get region labels (skip background at index 0)
    region_labels = [
        label.decode("utf-8") if isinstance(label, bytes) else label
        for label in atlas["labels"][1:]
    ]

    # Ensure exactly 90 regions
    if len(region_labels) > 90:
        region_labels = region_labels[:90]
        correlation_matrix = correlation_matrix[:90, :90]

    print(f"✓ Matrix: {correlation_matrix.shape}")
    print(
        f"✓ Range: [{correlation_matrix.min():.3f}, {correlation_matrix.max():.3f}]\n"
    )

    return correlation_matrix, region_labels


# ==============================================================================
# SYSTEM ASSIGNMENT
# ==============================================================================


def assign_functional_system(region_name: str) -> str:
    """
    Map AAL region to functional brain system.
    Based on neuroanatomy and Yeo 7-network parcellation.
    """
    name = region_name.lower()

    if any(x in name for x in ["occipital", "calcarine", "cuneus", "lingual"]):
        return "Visual"
    elif any(
        x in name for x in ["precentral", "postcentral", "paracentral", "rolandic"]
    ):
        return "Somatomotor"
    elif any(x in name for x in ["parietal", "supramarginal", "angular"]):
        return "Attention"
    elif any(x in name for x in ["frontal_sup", "frontal_mid", "cingulum_ant"]):
        return "ExecutiveControl"
    elif any(x in name for x in ["precuneus", "cingulum_post", "frontal_sup_medial"]):
        return "DefaultMode"
    elif any(x in name for x in ["hippocampus", "parahippocampal", "amygdala"]):
        return "Memory"
    elif any(x in name for x in ["insula", "frontal_inf_oper"]):
        return "Salience"
    elif any(x in name for x in ["temporal", "heschl"]):
        return "Language"
    else:
        return "Subcortical"


# ==============================================================================
# SAVE DATA
# ==============================================================================


def save_data(correlation_matrix, region_labels):
    """
    Save connectivity matrix, region labels, and system mappings.
    Creates three files required by analysis pipeline.
    """
    print("Step 5/5: Saving files...")

    # 1. Connectivity matrix
    df_matrix = pd.DataFrame(
        correlation_matrix, index=region_labels, columns=region_labels
    )
    matrix_path = DATA_DIR / "functional_connectivity_90x90.csv"
    df_matrix.to_csv(matrix_path)
    print(f"✓ {matrix_path}")

    # 2. Region labels
    systems = [assign_functional_system(label) for label in region_labels]
    labels_data = {
        "region_id": list(range(len(region_labels))),
        "region_name": region_labels,
        "system": systems,
    }
    df_labels = pd.DataFrame(labels_data)
    labels_path = DATA_DIR / "region_labels.csv"
    df_labels.to_csv(labels_path, index=False)
    print(f"✓ {labels_path}")

    # 3. System mappings
    network_map = {label: system for label, system in zip(region_labels, systems)}
    json_path = DATA_DIR / "learning_networks.json"
    with json_path.open("w") as f:
        json.dump(network_map, f, indent=2)
    print(f"✓ {json_path}")

    return systems


def print_summary(correlation_matrix, systems):
    """Print dataset summary and next steps."""
    print("\n" + "=" * 70)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 70)

    print("\nDATASET:")
    print("  Source: Development fMRI (Richardson et al., 2018)")
    print(f"  Matrix: {correlation_matrix.shape[0]}×{correlation_matrix.shape[1]}")
    print(f"  Range: [{correlation_matrix.min():.3f}, {correlation_matrix.max():.3f}]")

    print("\nSYSTEMS:")
    system_counts = pd.Series(systems).value_counts()
    for system, count in system_counts.items():
        print(f"  {system:20s}: {count:2d} regions")

    print("\nCITATION:")
    print("  Richardson, H., et al. (2018). Nature Communications, 9, 1027.")
    print("  Tzourio-Mazoyer, N., et al. (2002). NeuroImage, 15(1), 273-289.")

    print("\nReady to run: cd src && python main.py")
    print("=" * 70 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """Execute data download pipeline."""
    try:
        data = download_fmri_data()
        correlation_matrix, region_labels = compute_connectivity(data)
        systems = save_data(correlation_matrix, region_labels)
        print_summary(correlation_matrix, systems)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Run: pip install -r requirements.txt")
        print("  3. Try again (downloads are cached)")
        exit(1)


if __name__ == "__main__":
    main()
