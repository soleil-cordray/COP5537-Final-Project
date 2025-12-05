# Brain Networks for Learning: Technical Documentation

**Course**: COP5537 Network Optimization
**Project**: Network Analysis of Brain Connectivity
**Author**: Soleil Cordray
**Submission Date**: December 5, 2025

---

## Quick Start

```bash
# 1. Clone/download this repository
# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the brain connectivity data
python download_data.py

# 4. Run complete analysis pipeline
cd src && python main.py

# 5. View results
open results/figures/
```

**Expected Runtime**: ~10 minutes on standard hardware

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Project Structure](#project-structure)
5. [Running the Analysis](#running-the-analysis)
6. [Output Files](#output-files)
7. [Code Architecture](#code-architecture)
8. [Troubleshooting](#troubleshooting)
9. [Dataset Information](#dataset-information)
10. [Methodology](#methodology)

---

## Project Overview

This project applies network optimization algorithms to understand brain connectivity patterns that support learning and how mental health conditions disrupt these networks. Using a 90×90 functional connectivity matrix from resting-state fMRI, we implement:

- **Maximum flow analysis** to identify critical learning pathways
- **Centrality measures** (degree, betweenness, PageRank, closeness) to rank brain region importance
- **Mental health simulations** modeling depression, anxiety, and ADHD disruptions
- **Information spreading dynamics** to visualize attention propagation
- **Community detection** to discover functional brain systems

The complete implementation demonstrates network optimization concepts from COP5537 applied to a real-world neuroscience problem.

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (tested on 3.10-3.13)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk Space**: 2 GB free (for libraries + data + results)
- **OS**: macOS, Linux, or Windows

### Required Python Libraries
All dependencies are listed in `requirements.txt`:

```txt
# Core scientific computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Network analysis
networkx>=3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine learning (for clustering)
scikit-learn>=1.3.0

# Neuroimaging (for download_data.py)
nilearn>=0.10.0
nibabel>=5.0.0
```

---

## Installation Guide

### Step 1: Verify Python Installation

Check your Python version:

```bash
# macOS/Linux
python3 --version

# Windows
python --version
```

**Expected output**: `Python 3.X.Y` where X ≥ 8

### Step 2: Choose Installation Method

#### Option A: Using pip directly (Recommended for most users)

```bash
pip install -r requirements.txt
```

#### Option B: Using a virtual environment (Recommended for development)

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Windows:**
```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

Test that core libraries are accessible:

```python
python3 -c "import networkx, numpy, pandas, matplotlib; print('✓ All core libraries installed')"
```

### Step 4: Download Brain Connectivity Data

```bash
python download_data.py
```

This script:
- Downloads resting-state fMRI data (~200 MB, takes 2-5 minutes)
- Computes the 90×90 functional connectivity matrix
- Generates region labels and system assignments
- Saves three files to `data/`:
  - `functional_connectivity_90x90.csv`
  - `region_labels.csv`
  - `learning_networks.json`

---

## Project Structure

```
brain-networks-learning/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── download_data.py             # Data acquisition script
├── .gitignore                   # Git exclusions (includes __pycache__)
├── pyrightconfig.json           # Python type-checking config
│
├── data/                        # Created by download_data.py
│   ├── functional_connectivity_90x90.csv   # 90×90 correlation matrix
│   ├── region_labels.csv                    # Region metadata
│   └── learning_networks.json               # System mappings
│
├── src/                         # Analysis modules
│   ├── main.py                  # Master pipeline script
│   ├── data_loader.py           # Data loading & graph construction
│   ├── explore.py               # Network statistics & visualization
│   ├── learning_pathways.py     # Max-flow/min-cut analysis
│   ├── hub_analysis.py          # Centrality computations
│   ├── mental_health_sim.py     # Mental health disruption models
│   ├── attention_dynamics.py    # Information spreading simulation
│   └── functional_systems.py    # Community detection
│
├── results/                     # Created by analysis scripts
│   ├── figures/                 # Publication-quality visualizations (PNG)
│   │   ├── brain_network_learning.png
│   │   ├── connectivity_matrix.png
│   │   ├── degree_distribution.png
│   │   └── functional_communities.png
│   │
│   └── data/                    # Numerical results (CSV/JSON)
│       ├── network_statistics.json
│       ├── learning_hubs.csv
│       ├── learning_pathways.json
│       ├── centralities_by_condition.csv
│       ├── mental_health_summary.csv
│       └── attention_dynamics.csv
│
└── report.pdf                   # 4-5 page project report (submission)
```

---

## Running the Analysis

### Complete Pipeline (Recommended)

Execute all analyses in sequence:

```bash
cd src
python main.py
```

**What this does**:
1. Loads 90×90 connectivity matrix and constructs weighted graph
2. Computes basic network statistics (density, clustering, path length)
3. Runs max-flow analysis for learning pathways (attention → memory)
4. Calculates centrality measures for all 90 brain regions
5. Simulates mental health disruptions (depression, anxiety, ADHD)
6. Models information spreading dynamics
7. Detects functional communities via modularity optimization
8. Generates all figures and exports numerical results

**Runtime**: ~10 minutes
**Output**: 4 figures + 6 data files in `results/`

### Individual Module Execution

Run specific analyses independently:

```bash
# Network exploration (density, clustering, visualizations)
python src/explore.py

# Learning pathways (max flow from attention to memory systems)
python src/learning_pathways.py

# Hub identification (centrality rankings)
python src/hub_analysis.py

# Mental health simulations
python src/mental_health_sim.py

# Attention spreading dynamics
python src/attention_dynamics.py

# Community detection
python src/functional_systems.py
```

**Note**: All modules can run independently; they each load data and save their own outputs.

---

## Output Files

### Figures (`results/figures/`)

All visualizations are 300 DPI PNG files suitable for reports:

1. **connectivity_matrix.png** (5.4 MB)
   - 90×90 heatmap showing correlation strengths
   - Red = positive correlation (co-activation)
   - Blue = negative correlation (anti-correlation)

2. **brain_network_learning.png** (1.2 MB)
   - Spring-layout network visualization
   - Node size ∝ degree (connectivity)
   - Node color = functional system

3. **degree_distribution.png** (850 KB)
   - Histogram showing distribution of node degrees
   - Most nodes have 70-85 connections (highly connected network)

4. **functional_communities.png** (1.8 MB)
   - Community structure visualization
   - Colors indicate detected functional systems
   - Modularity Q value displayed in title

### Data Files (`results/data/`)

All numerical results in CSV or JSON format:

1. **network_statistics.json**
   ```json
   {
     "n_nodes": 90,
     "n_edges": 3346,
     "density": 0.835,
     "avg_clustering": 0.466,
     "avg_path_length_largest_component": 1.164,
     "is_connected": true
   }
   ```

2. **learning_hubs.csv**
   - 90 rows × 6 columns
   - Columns: region, degree, betweenness, closeness, pagerank, system
   - Ranked by centrality measures

3. **learning_pathways.json**
   - Max flow value from attention systems to memory systems
   - Lists of source and target regions

4. **centralities_by_condition.csv**
   - 360 rows (90 regions × 4 conditions)
   - Tracks how centrality changes under mental health disruptions

5. **mental_health_summary.csv**
   - 4 rows (Healthy, Depression, Anxiety, ADHD)
   - Learning efficiency ratios and PageRank statistics

6. **attention_dynamics.csv**
   - Temporal sequence of region activation
   - Simulates information spreading from visual cortex

---

## Code Architecture

### Module Responsibilities

#### `data_loader.py`
**Purpose**: Central data management and graph construction

**Key Functions**:
- `load_brain_network()` → Returns `BrainNetworkData` object
- `build_graph_from_matrix()` → Converts correlation matrix to NetworkX graph

**Graph Construction Logic**:
```python
# Edges are added only if |correlation| ≥ threshold (default 0.2)
# Edge weight = original signed correlation value
# Self-loops excluded
# Symmetry enforced: (i,j) weight = (j,i) weight
```

**Why symmetry enforcement?**
Correlation matrices should be symmetric, but numerical noise can create slight differences. We average each pair: `matrix = (matrix + matrix.T) / 2`

#### `explore.py`
**Purpose**: Basic network analysis and visualization

**Computed Metrics**:
- Density: fraction of possible edges present
- Average clustering coefficient
- Average shortest path length (on largest connected component)
- Degree distribution

**Generates**:
- Connectivity heatmap
- Degree distribution histogram
- Network overview with hubs highlighted

#### `learning_pathways.py`
**Purpose**: Max-flow/min-cut analysis

**Algorithm**: Edmonds-Karp (NetworkX implementation)

**Approach**:
1. Create super-source node connected to all attention regions
2. Create super-sink node connected from all memory regions
3. Run max-flow from super-source to super-sink
4. Result = maximum information capacity through attention→memory pathway

**Edge Capacities**: We use absolute correlation values as capacities (negative correlations become positive for flow purposes)

#### `hub_analysis.py`
**Purpose**: Centrality computations for all 90 regions

**Four Centrality Measures**:
1. **Degree**: `deg(v)` = number of neighbors
2. **Betweenness**: fraction of shortest paths passing through node
3. **Closeness**: `1 / Σ dist(v, u)` for all u
4. **PageRank**: Google's algorithm adapted to brain networks

**Interpretation**:
- High degree = highly connected hub
- High betweenness = critical bottleneck (removing it disconnects network)
- High closeness = efficiently reaches all other regions
- High PageRank = connected to other important regions

#### `mental_health_sim.py`
**Purpose**: Simulate connectivity disruptions

**Disruption Models** (based on neuroscience literature):

**Depression**:
```python
DMN internal connectivity *= 1.3        # Hyperactive rumination
Executive control *= 0.8                # Weakened cognitive control
DMN-Executive anticorrelation *= 1.1    # Reduced suppression
```

**Anxiety**:
```python
Salience network *= 1.3                 # Heightened threat detection
Salience-Attention edges *= 1.2         # Attention hijacked by threats
Executive control *= 0.85               # Reduced top-down regulation
```

**ADHD**:
```python
Executive control *= 0.7                # Severe executive dysfunction
Attention networks *= 0.7               # Poor attentional stability
DMN-Executive anticorrelation *= 1.1    # DMN intrudes during tasks
```

After applying disruptions, centralities are recomputed to measure impact.

#### `attention_dynamics.py`
**Purpose**: Simulate information spreading through the network

**Algorithm**:
- Start with seed regions (visual cortex)
- At each time step, activate neighbors across strong connections (threshold = 0.25)
- Track temporal sequence of activation
- Saves CSV with (time_step, region, system) entries

**Use Case**: Models how reading-related information propagates from visual cortex through attention systems to memory regions.

#### `functional_systems.py`
**Purpose**: Community detection via modularity optimization

**Algorithm**: Greedy modularity (Clauset-Newman-Moore)

**Modularity Q**: Measures quality of partition
- Q > 0.3 = significant community structure
- Q > 0.5 = strong community structure
- Our result: Q ≈ 0.06 (highly integrated network with weak modularity)

**Why low modularity?**
Brain networks, especially at rest, are highly integrated. The low modularity reflects dense inter-system connectivity rather than isolated clusters. This is consistent with neuroscience findings that cognitive functions require cross-system coordination.

---

## Troubleshooting

### Common Installation Issues

#### Issue: `pip: command not found`
**Solution**:
```bash
# macOS/Linux
python3 -m ensurepip --upgrade
python3 -m pip install -r requirements.txt

# Windows
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

#### Issue: `ERROR: externally-managed-environment` (macOS with Homebrew)
**Solution**: Use a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Why this happens**: Homebrew-managed Python prevents system-wide pip installations to avoid conflicts.

#### Issue: SSL Certificate Error (during `download_data.py`)
**Symptoms**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**: Already handled in `download_data.py`
```python
# Applied automatically on SSL errors
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

If still failing, download manually:
1. Go to https://www.humanconnectome.org/
2. Download AAL atlas connectivity matrix
3. Place in `data/functional_connectivity_90x90.csv`

#### Issue: Import Error - `ModuleNotFoundError: No module named 'XXX'`
**Solution**:
```bash
# Reinstall specific missing library
pip install <library-name>

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

### Runtime Issues

#### Issue: `FileNotFoundError: data/functional_connectivity_90x90.csv`
**Cause**: Data not downloaded yet

**Solution**:
```bash
python download_data.py
```

#### Issue: Script hangs during max-flow computation
**Cause**: Very large graphs can be slow

**Solution**: This should not occur with 90 nodes. If it does:
1. Check available RAM (need ~2 GB free)
2. Close other applications
3. If still hanging after 5 minutes, press Ctrl+C and restart

#### Issue: Figures not saving
**Cause**: `results/figures/` directory doesn't exist

**Solution**: Already handled in code
```python
# Each script creates directories if needed
out_path.parent.mkdir(parents=True, exist_ok=True)
```

If still failing, manually create:
```bash
mkdir -p results/figures results/data
```

### Performance Optimization

**Slow execution?** Try:
1. Reduce connectivity threshold in `data_loader.py`:
   ```python
   brain = load_brain_network(weight_threshold=0.3)  # Default is 0.2
   ```
   Higher threshold = fewer edges = faster computation

2. Skip expensive analyses:
   ```bash
   # Run only essential analyses
   python src/explore.py
   python src/hub_analysis.py
   ```

3. Use PyPy (alternative Python interpreter, faster for some operations):
   ```bash
   pypy3 -m pip install -r requirements.txt
   pypy3 src/main.py
   ```

---

## Dataset Information

### Source

**Development fMRI Dataset** (Richardson et al., 2018)
- Available through Nilearn library
- Part of Human Connectome Project initiative
- Resting-state fMRI from healthy adults

### Anatomical Atlas

**AAL (Automated Anatomical Labeling)**
- 90 cortical and subcortical regions
- Widely used standard in neuroimaging
- Reference: Tzourio-Mazoyer et al. (2002), NeuroImage

### Functional Systems

Regions are mapped to 9 functional systems based on neuroscience literature:

1. **Visual** (12 regions): Occipital cortex, calcarine, cuneus
2. **Somatomotor** (6 regions): Pre/postcentral gyrus, paracentral lobule
3. **Attention** (8 regions): Parietal cortex (superior, inferior, supramarginal, angular)
4. **ExecutiveControl** (6 regions): Dorsolateral/medial prefrontal, anterior cingulate
5. **DefaultMode** (2 regions): Posterior cingulate, medial prefrontal
6. **Memory** (6 regions): Hippocampus, parahippocampal, amygdala
7. **Salience** (4 regions): Insula, inferior frontal
8. **Language** (10 regions): Temporal cortex, Heschl's gyrus
9. **Subcortical** (36 regions): Thalamus, basal ganglia, other deep structures

### Connectivity Matrix Properties

- **Size**: 90 × 90 = 8,100 entries
- **Values**: Pearson correlations, range [-1, +1]
- **Symmetry**: Enforced (brain connectivity is bidirectional)
- **Diagonal**: All 1.0 (self-correlation)
- **Preprocessing**: Standardization, motion correction, physiological noise removal (done by nilearn)

### Graph Construction

From correlation matrix to graph:

```python
# Thresholding: only edges with |correlation| ≥ 0.2
# Why? Remove weak/spurious connections, focus on strong pathways
# Result: 3,346 edges out of 4,005 possible (83.5% density)

# Edge weights: Original signed correlation values
# Positive weight = co-activation (regions work together)
# Negative weight = anti-correlation (one active → other suppressed)
```

**High density interpretation**: Brain is highly integrated, not modular. Most regions communicate with most others, enabling flexible cognition.

---

## Methodology

### Network Optimization Concepts Applied

#### 1. Maximum Flow / Minimum Cut
**Algorithm**: Edmonds-Karp (augmenting paths)
**Application**: Measure information capacity from attention systems to memory systems
**Result**: Identifies bottleneck regions limiting learning

**Course Connection**: Topics 8-9 (Network Flow)
- Flow conservation constraints
- Capacity constraints
- Max-flow min-cut theorem: maximum flow value = minimum cut capacity

#### 2. Centrality Measures
**Algorithms**: Degree, betweenness, closeness, PageRank
**Application**: Rank brain regions by importance
**Result**: Distinguishes learning hubs from peripheral regions

**Course Connection**: Topics 10-11 (Centralities, PageRank & HITS)
- Degree centrality: Count connections
- Betweenness: Fraction of shortest paths through node
- Closeness: Average distance to all other nodes
- PageRank: Iterative importance via neighbors

#### 3. Graph Clustering
**Algorithm**: Greedy modularity optimization (Louvain-style)
**Application**: Discover functional brain systems
**Result**: Identifies communities of highly interconnected regions

**Course Connection**: Topic 12 (Graph Clustering)
- Modularity Q metric
- Community detection algorithms
- Spectral clustering concepts

#### 4. Graph Properties
**Metrics**: Density, clustering coefficient, path length
**Application**: Characterize brain network topology
**Result**: Confirms small-world properties (high clustering + short paths)

**Course Connection**: Topics 1-7 (Graph Fundamentals)
- Degree distribution
- Clustering coefficient
- Shortest paths (Dijkstra's algorithm concepts)
- Connected components

### Mental Health Disruption Modeling

Based on neuroscience literature documenting connectivity changes:

**Depression** ([Kaiser et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25599391/)):
- DMN hyperconnectivity → rumination
- Reduced executive control → difficulty concentrating
- Weakened DMN-executive anticorrelation → DMN intrudes during tasks

**Anxiety** ([Paulus & Stein, 2006](https://pubmed.ncbi.nlm.nih.gov/17046034/)):
- Salience network hyperactivity → heightened threat detection
- Attention bias toward threats → reduced learning focus
- Executive dysfunction → poor emotion regulation

**ADHD** ([Castellanos et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18629236/)):
- Executive/attention network deficits → poor sustained attention
- Reduced DMN suppression → mind wandering during tasks
- Reward system differences → motivation challenges

These disruptions are simulated by scaling edge weights, then measuring the impact on learning pathway capacity and hub centralities.

---

## Additional Resources

### References

**Course Materials**:
- COP5537 Network Optimization Lecture Notes
- Textbook: "Network Flows" by Ahuja, Magnanti, and Orlin

**Neuroscience Papers**:
- Finn et al. (2015). "Functional connectome fingerprinting." Nature Neuroscience
- Yeo et al. (2011). "Organization of the human cerebral cortex." Journal of Neurophysiology
- Kaiser et al. (2015). "Large-scale network dysfunction in major depressive disorder." JAMA Psychiatry

**Software Documentation**:
- NetworkX: https://networkx.org/documentation/stable/
- Nilearn: https://nilearn.github.io/stable/
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/contents.html

### Future Extensions

**Immediate next steps**:
1. Download full Human Connectome Project dataset (1,200 subjects)
2. Implement task-based connectivity analysis
3. Add statistical validation (permutation testing)
4. Create interactive visualizations (Plotly, NetworkX with D3.js export)

**Long-term directions**:
1. Personalized network medicine: Individual-level predictions
2. Longitudinal tracking: How does therapy/medication restore connectivity?
3. Intervention design: Which connections should training target?
4. Cross-species comparison: How do human learning networks compare to other species?

---

**Last Updated**: December 5, 2025
**Version**: 1.0 (Final Submission)
