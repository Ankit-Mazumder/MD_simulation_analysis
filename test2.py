import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. DATA PREPARATION (GB_4 is a feature, Target is Fixed)
# ==========================================
def prepare_data(edge_file, node_file, graph_file, target_value):
    print(f"Loading raw data... Target Delta G set to: {target_value}")
    edges_df = pd.read_csv(edge_file)
    nodes_df = pd.read_csv(node_file)
    graph_df = pd.read_csv(graph_file)

    # --- Clean & Fill NaNs (Nodes) ---
    numeric_node_cols = ['Z', 'mass', 'rmsd', 'rmsf']
    for col in numeric_node_cols:
        if col in nodes_df.columns:
            nodes_df[col] = pd.to_numeric(nodes_df[col], errors='coerce').fillna(0)

    # ============================================================
    # LEVEL 1: Angle Feature Engineering (Sin/Cos Encoding)
    # ============================================================
    
    # 1. Clean Distance first
    if 'distance' in edges_df.columns:
        edges_df['distance'] = pd.to_numeric(edges_df['distance'], errors='coerce').fillna(0)

    # 2. Transform Angle
    if 'angle' in edges_df.columns:
        print("Applying Level 1: Transforming angles to Sin/Cos encoding...")
        # Force numeric
        raw_angles = pd.to_numeric(edges_df['angle'], errors='coerce').fillna(0)
        
        # Convert Degrees to Radians (Crucial step)
        # Assuming input is in degrees. If already radians, remove np.deg2rad
        angles_rad = np.deg2rad(raw_angles)
        
        # Create the new features
        edges_df['angle_sin'] = np.sin(angles_rad)
        edges_df['angle_cos'] = np.cos(angles_rad)
        
        # Drop the original raw angle column to avoid cyclic noise
        edges_df.drop(columns=['angle'], inplace=True)
    else:
        print("Warning: 'angle' column not found. Skipping angle encoding.")

    # ============================================================

    # Ensure global features are numeric (Everything except frame)
    for col in graph_df.columns:
        if col != 'frame':
            graph_df[col] = pd.to_numeric(graph_df[col], errors='coerce').fillna(0)

    # --- One-Hot Encoding ---
    nodes_df = pd.get_dummies(nodes_df, columns=['element', 'chain'])
    edges_df = pd.get_dummies(edges_df, columns=['bond_type'])

    # --- Identify Feature Columns ---
    # Note: This logic AUTOMATICALLY picks up 'angle_sin' and 'angle_cos' 
    # because they are now columns in edges_df and are not 'frame', 'atom_i', etc.
    node_feature_cols = [col for col in nodes_df.columns if col not in ['frame', 'atom_index']]
    edge_feature_cols = [col for col in edges_df.columns if col not in ['frame', 'atom_i', 'atom_j']]
    global_feature_cols = [col for col in graph_df.columns if col != 'frame']
    
    print(f"Features: {len(node_feature_cols)} Node, {len(edge_feature_cols)} Edge, {len(global_feature_cols)} Global.")
    # Debug print to confirm angles are there
    if 'angle_sin' in edge_feature_cols:
        print("--> Confirmed: 'angle_sin' and 'angle_cos' are included in edge features.")

    # --- SCALING ---
    print("Applying StandardScaler to FEATURES only...")
    
    # 1. Scale Node Features
    node_scaler = StandardScaler()
    nodes_df[node_feature_cols] = node_scaler.fit_transform(nodes_df[node_feature_cols].astype(np.float32))
    
    # 2. Scale Edge Features
    edge_scaler = StandardScaler()
    edges_df[edge_feature_cols] = edge_scaler.fit_transform(edges_df[edge_feature_cols].astype(np.float32))

    # 3. Scale Global Features
    global_scaler = StandardScaler()
    scaled_global_feats = global_scaler.fit_transform(graph_df[global_feature_cols].astype(np.float32))
    global_map = dict(zip(graph_df['frame'], scaled_global_feats))

    # 4. TARGET PREPARATION (Manual Scaling)
    TARGET_SCALE_FACTOR = 1000.0 
    scaled_target_value = target_value / TARGET_SCALE_FACTOR

    # --- Build Data Objects ---
    all_data = []
    frames = sorted(nodes_df['frame'].unique())
    
    print(f"Building graph objects for {len(frames)} frames...")
    
    for frame in frames:
        frame_nodes = nodes_df[nodes_df['frame'] == frame].sort_values('atom_index')
        frame_edges = edges_df[edges_df['frame'] == frame]

        u_feat = global_map.get(frame) 

        if u_feat is None: continue
            
        x = torch.tensor(frame_nodes[node_feature_cols].values, dtype=torch.float)
        edge_attr = torch.tensor(frame_edges[edge_feature_cols].values, dtype=torch.float)
        edge_index = torch.tensor(frame_edges[['atom_i', 'atom_j']].values.T, dtype=torch.long)
        
        # Assign the SAME target value to every frame
        y = torch.tensor([scaled_target_value], dtype=torch.float)
        
        # Add 'u' (Global Features)
        u = torch.tensor(u_feat, dtype=torch.float).unsqueeze(0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
        all_data.append(data)

    print("Data preparation complete.")
    return all_data, len(node_feature_cols), len(edge_feature_cols), len(global_feature_cols), TARGET_SCALE_FACTOR
# ==========================================
# 2. DATASET & COLLATOR
# ==========================================
class TemporalGraphWindowDataset(Dataset):
    def __init__(self, data_list, window_size):
        self.data_list = data_list
        self.window_size = window_size

    def __len__(self):
        return len(self.data_list) - self.window_size + 1

    def __getitem__(self, idx):
        input_window = self.data_list[idx : idx + self.window_size]
        target = self.data_list[idx + self.window_size - 1].y
        return input_window, target

def collate_fn(batch):
    input_windows, targets = zip(*batch)
    graphs_list = [graph for window in input_windows for graph in window]
    batched_graphs = Batch.from_data_list(graphs_list)
    targets_tensor = torch.stack(targets)
    return batched_graphs, targets_tensor

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GraphTransformer(nn.Module):
    def __init__(self, node_f, edge_f, global_f, hidden_dim, out_dim, 
                 transformer_heads, transformer_layers, window_size):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        self.conv1 = GINEConv(nn.Sequential(Linear(node_f, hidden_dim), nn.ReLU(), Linear(hidden_dim, hidden_dim)), edge_dim=edge_f)
        self.conv2 = GINEConv(nn.Sequential(Linear(hidden_dim, hidden_dim), nn.ReLU(), Linear(hidden_dim, hidden_dim)), edge_dim=edge_f)
        self.conv3 = GINEConv(nn.Sequential(Linear(hidden_dim, hidden_dim), nn.ReLU(), Linear(hidden_dim, hidden_dim)), edge_dim=edge_f)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=window_size)
        
        self.fusion_mlp = nn.Sequential(
            Linear(hidden_dim + global_f, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=transformer_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=False,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.fc_out = Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch, u = data.x, data.edge_index, data.edge_attr, data.batch, data.u
        
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr).relu()
        
        h_graph = global_mean_pool(x, batch) 
        
        combined = torch.cat([h_graph, u], dim=1)
        h_fused = self.fusion_mlp(combined)

        B = h_fused.shape[0] // self.window_size 
        seq = h_fused.view(B, self.window_size, self.hidden_dim)
        seq = seq.transpose(0, 1) 
        
        seq_pos = self.pos_encoder(seq)
        tf_out = self.transformer_encoder(seq_pos)
        
        last_step_out = tf_out[-1, :, :]
        out = self.fc_out(last_step_out)
        return out

# ==========================================
# 4. PLOTTING FUNCTION (Includes Loss Curve)
# ==========================================
def generate_plots(model, test_loader, target_scale_factor, device, train_history, val_history, save_dir='results_plots'):
    print("\nGenerating analysis plots using BEST Saved Model...")
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batched_graphs, targets in test_loader:
            batched_graphs = batched_graphs.to(device)
            out = model(batched_graphs)
            all_preds.extend(out.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    # Manually Rescale Back
    real_preds = np.array(all_preds) * target_scale_factor
    real_targets = np.array(all_targets) * target_scale_factor
    
    # --- Plot 1: Trajectory ---
    plt.figure(figsize=(12, 5))
    subset = min(200, len(real_targets))
    plt.plot(real_targets[:subset], label='Experimental ΔG', color='black', linewidth=2)
    plt.plot(real_preds[:subset], label='Predicted ΔG', color='red', linestyle='--', alpha=0.7)
    plt.title("Prediction Stability over Trajectory")
    plt.xlabel("Time Window")
    plt.ylabel("Delta G (cal/mol)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'trajectory_dg.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Parity ---
    plt.figure(figsize=(6, 6))
    plt.scatter(real_targets, real_preds, alpha=0.5, s=10, color='blue')
    min_val = min(real_targets.min(), real_preds.min()) - 100
    max_val = max(real_targets.max(), real_preds.max()) + 100
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    r2 = r2_score(real_targets, real_preds)
    rmse = np.sqrt(mean_squared_error(real_targets, real_preds))
    plt.title(f"Parity Plot\nR²={r2:.3f}, RMSE={rmse:.2f}")
    plt.xlabel("Actual ΔG")
    plt.ylabel("Predicted ΔG")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'parity_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Error Hist ---
    residuals = real_preds - real_targets
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color='purple')
    plt.title("Error Distribution")
    plt.xlabel("Error (Pred - Actual)")
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 4: Loss Curve (NEW) ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Train RMSE', color='blue')
    plt.plot(val_history, label='Validation RMSE', color='orange')
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (cal/mol)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {save_dir}/ directory.")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

# --- Hyperparameters ---
WINDOW_SIZE = 5         
BATCH_SIZE = 64         
HIDDEN_DIM = 128        
OUT_DIM = 1
EPOCHS = 100           
LEARNING_RATE = 0.001   
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 3
TEST_SPLIT = 0.2

# --- INPUTS ---
edge_file = 'protein_graph_edges_ECIF_RF_GB_allatoms6.csv'
node_file = 'protein_graph_nodes_ECIF_RF_GB_allatoms6.csv'
graph_file = 'protein_graph_global_ECIF_RF_GB_allatoms6.csv'

# !!! SET YOUR EXPERIMENTAL TARGET HERE !!!
EXPERIMENTAL_DELTA_G = -10667.0  

# Prepare Data
all_data, node_f, edge_f, global_f, target_scale_factor = prepare_data(
    edge_file, node_file, graph_file, target_value=EXPERIMENTAL_DELTA_G
)

# --- Split & Loaders ---
test_size = int(len(all_data) * TEST_SPLIT)
train_data_list = all_data[:-test_size]
test_data_list = all_data[-test_size:]

train_dataset = TemporalGraphWindowDataset(train_data_list, WINDOW_SIZE)
test_dataset = TemporalGraphWindowDataset(test_data_list, WINDOW_SIZE)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    num_workers=4, pin_memory=True, persistent_workers=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    num_workers=4, pin_memory=True, persistent_workers=True
)

print(f"\nTrain windows: {len(train_dataset)} | Test windows: {len(test_dataset)}")

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GraphTransformer(
    node_f=node_f, edge_f=edge_f, global_f=global_f,
    hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
    transformer_heads=TRANSFORMER_HEADS, transformer_layers=TRANSFORMER_LAYERS,
    window_size=WINDOW_SIZE
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
loss_fn = nn.MSELoss()

# ... (Previous setup code remains the same) ...

# ==========================================
# CORRECTED LOOP WITH WARM-UP
# ==========================================

# New Hyperparameter: Ignore "Best" scores during these epochs
WARMUP_EPOCHS = 20  

print(f"\nStarting training to predict ΔG = {EXPERIMENTAL_DELTA_G}...")
print(f"Warm-up period: First {WARMUP_EPOCHS} epochs (Model saving disabled).")

best_val_loss = float('inf')
best_model_path = 'best_model_dg.pth'

train_rmse_history = []
val_rmse_history = []

for epoch in range(1, EPOCHS + 1):
    # --- TRAIN ---
    model.train()
    total_loss = 0
    
    for batched_graphs, targets in train_loader:
        batched_graphs = batched_graphs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        out = model(batched_graphs)
        loss = loss_fn(out.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    current_real_rmse = math.sqrt(avg_loss) * target_scale_factor
    train_rmse_history.append(current_real_rmse)

    # --- VALIDATION ---
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batched_graphs, targets in test_loader:
            batched_graphs = batched_graphs.to(device)
            targets = targets.to(device)
            out = model(batched_graphs)
            loss = loss_fn(out.squeeze(), targets.squeeze())
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_real_rmse = math.sqrt(avg_test_loss) * target_scale_factor
    val_rmse_history.append(test_real_rmse)
    
    # Update Scheduler
    scheduler.step(avg_test_loss)

    # --- SAVING LOGIC (WITH WARM-UP) ---
    # Only consider saving if we are past the warm-up period
    if epoch > WARMUP_EPOCHS:
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch:03d} | * SAVED BEST * | Val RMSE: {test_real_rmse:.2f}")
        
        elif epoch % 5 == 0:
             print(f"Epoch {epoch:03d} | Train RMSE: {current_real_rmse:.2f} | Val RMSE: {test_real_rmse:.2f}")
    else:
        # During warm-up, just print status
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | [Warm-up] Train RMSE: {current_real_rmse:.2f} | Val RMSE: {test_real_rmse:.2f}")

# ... (Verification and Plotting code remains the same) ...

# --- Final Plotting ---
print(f"Loading best model for plotting...")
try:
    model.load_state_dict(torch.load(best_model_path))
    generate_plots(model, test_loader, target_scale_factor, device, train_rmse_history, val_rmse_history)
except Exception as e:
    print(f"Error loading model: {e}")

# # --- Final Plotting ---
# print(f"\nLoading best model from {best_model_path}...")
# model.load_state_dict(torch.load(best_model_path))

# Pass the history lists to the plotting function
generate_plots(model, test_loader, target_scale_factor, device, train_rmse_history, val_rmse_history)