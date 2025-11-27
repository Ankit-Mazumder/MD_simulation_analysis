Spatio-Temporal Graph Transformer for Molecular Dynamics

A hybrid Deep Learning framework combining Graph Neural Networks (GINEConv) with Temporal Transformers to predict thermodynamic properties (specifically Gibbs Free Energy, $\Delta G$) from Molecular Dynamics (MD) simulation trajectories.

This model captures both the spatial geometry of atoms within a single frame and the temporal evolution of the system across a sliding time window.

🧠 Model Architecture

The model processes sequential windows of molecular graphs (frames) to predict a single scalar value for the window.

Data Flow Pipeline

Input: A window of $T$ frames (default $T=5$). Each frame is a graph $G = (V, E)$.

Spatial Encoding (GNN): * 3 layers of GINEConv (Graph Isomorphism Network with Edge features).

Extracts atomic interactions and local geometry.

Global Pooling: Aggregates node features into a single vector per frame.

Feature Fusion: Concatenates Graph Embeddings with Global System Features (Temperature, Pressure, etc.).

Temporal Encoding (Transformer):

Positional Encoding: Injects time-step order information.

Transformer Encoder: 3-Layer Multi-Head Self-Attention mechanism to learn trajectory dynamics.

Readout: Regression head predicts $\Delta G$ based on the final state of the sequence.

📂 Data Requirements

The script expects three CSV files in the root directory.

1. protein_graph_nodes_...csv

Contains atomic features.

Required Columns: frame, atom_index, Z (Atomic Number), mass, rmsd, rmsf, element (categorical), chain (categorical).

2. protein_graph_edges_...csv

Contains bond/interaction features.

Required Columns: frame, atom_i, atom_j, bond_type (categorical), distance, angle (degrees).

Note: The code automatically converts angle into sin(angle) and cos(angle) components.

3. protein_graph_global_...csv

Contains system-wide variables.

Required Columns: frame, plus any global features (e.g., Temperature, PotentialEnergy, GB_4).

🛠️ Installation

Ensure you have a Python environment (3.8+) with the following dependencies:

# Core DL Libraries
pip install torch torchvision
pip install torch-geometric

# Data Handling & Math
pip install pandas numpy scikit-learn

# Plotting
pip install matplotlib seaborn



Note: For torch-geometric, ensure you install the scatter/sparse binaries matching your CUDA version.

🚀 Usage

Configure Paths:
Open the script and edit the file paths in the MAIN EXECUTION section:

edge_file = 'path/to/edges.csv'
node_file = 'path/to/nodes.csv'
graph_file = 'path/to/global.csv'



Set Target:
Define your experimental $\Delta G$ value:

EXPERIMENTAL_DELTA_G = -10667.0 



Run the Training:

python model_training.py



Hyperparameters

Key parameters can be adjusted at the top of the MAIN EXECUTION block:

WINDOW_SIZE: Number of frames to look back (default: 5).

HIDDEN_DIM: Dimension of internal embeddings (default: 128).

TRANSFORMER_LAYERS: Depth of the temporal encoder (default: 3).

WARMUP_EPOCHS: Epochs to ignore for model saving (default: 20).

📊 Outputs & Visualization

Upon completion, the script creates a results_plots/ directory containing:

trajectory_dg.png: A time-series plot comparing Predicted vs. Experimental $\Delta G$ across the test trajectory.

parity_plot.png: Scatter plot showing correlation ($R^2$ and RMSE) between ground truth and predictions.

loss_curve.png: Training and Validation RMSE evolution over epochs.

error_distribution.png: Histogram of residual errors.

The script also saves the best model weights to best_model_dg.pth.

🔬 Feature Engineering Details

Angle Transformation: Raw angles (degrees) are automatically converted to cyclic encodings ($\sin(\theta), \cos(\theta)$) to preserve geometric continuity.

Scaling: * Node/Edge/Global features are scaled using StandardScaler.

The Target $\Delta G$ is scaled by a factor of 1000.0 for numerical stability during backpropagation.

Tokenization: Each frame in the window is treated as a "token" in the Transformer sequence, analogous to words in an NLP sentence.

📜 License

MIT License
