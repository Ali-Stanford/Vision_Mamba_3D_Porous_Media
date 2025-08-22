import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
from typing import Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------

def r2_score_custom(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def format_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - (h * 3600 + m * 60)
    return f"{h}h {m}m {s:.1f}s"

class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(pred, target) + self.eps)

# -------------------------
# 3D CNN Regressor
# -------------------------

class Permeability3DCNNRegressor(nn.Module):
    def __init__(self, in_chans: int = 1, drop_p: float = 0.7):
        super().__init__()
        filters = [16, 32, 64, 128, 256, 512, 1024]
        kernels = [2,  2,  2,   2,   2,   2,    1]
        strides = [2,  2,  2,   2,   2,   2,    2]

        convs = []
        c_in = in_chans
        for c_out, k, s in zip(filters, kernels, strides):
            convs += [
                nn.Conv3d(c_in, c_out, kernel_size=k, stride=s, padding=0, bias=False),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.features = nn.Sequential(*convs)

        self.regressor = nn.Sequential(
            nn.Flatten(),                      # [B, 1024*1*1*1] -> [B, 1024]
            nn.Linear(1024, 512, bias=False),  nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(p=drop_p),
            nn.Linear(512, 256, bias=False),   nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p=drop_p),
            nn.Linear(256, 1)                  # scalar (logit)
        )

    def forward(self, x):
        x = self.features(x)
        y = self.regressor(x).squeeze(-1)      # [B], logits
        #y = torch.sigmoid(y)                   # normalize to [0,1]
        return y

# -------------------------
# Dataset
# -------------------------

class CubeDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.hf = None
        self.cubes = None
        self.perms = None

    def _ensure_open(self):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')
            self.cubes = self.hf['cubes']   # [N, 64,64,64]
            self.perms = self.hf['perms']   # [N]

    def __len__(self):
        self._ensure_open()
        return self.perms.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        cube = self.cubes[idx]                               # (64,64,64)
        perm = float(self.perms[idx])                        # scalar permeability (physical units)
        cube = torch.from_numpy(cube).unsqueeze(0).float()   # [1,64,64,64]
        perm = torch.tensor(perm, dtype=torch.float32)       # []
        return cube, perm

    def __del__(self):
        if getattr(self, "hf", None) is not None:
            try:
                self.hf.close()
            except Exception:
                pass

# -------------------------
# Splits
# -------------------------

def load_or_create_splits(N, n_train, n_val, seed=0, prefix=Path(".")):
    train_p = prefix / "split_train_idx.npy"
    val_p   = prefix / "split_val_idx.npy"
    test_p  = prefix / "split_test_idx.npy"

    def valid_loaded(arr, expected_len):
        return arr is not None and arr.ndim == 1 and len(arr) == expected_len and arr.max() < N and arr.min() >= 0

    if train_p.exists() and val_p.exists() and test_p.exists():
        train_idx = np.load(train_p); val_idx = np.load(val_p); test_idx = np.load(test_p)
        if (len(train_idx) + len(val_idx) + len(test_idx) == N and
            len(np.intersect1d(train_idx, val_idx)) == 0 and
            len(np.intersect1d(train_idx, test_idx)) == 0 and
            len(np.intersect1d(val_idx, test_idx)) == 0 and
            valid_loaded(train_idx, n_train) and valid_loaded(val_idx, n_val)):
            print("Loaded existing train/val/test split indices from disk.")
            return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)
        else:
            print("Existing split files are invalid or out-of-date. Regenerating...")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]
    np.save(train_p, train_idx); np.save(val_p, val_idx); np.save(test_p, test_idx)
    print(f"Saved split indices to {train_p.name}, {val_p.name}, {test_p.name}.")
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)

# -------------------------
# Training
# -------------------------

def main():
    torch.backends.cudnn.benchmark = True

    # GPU selection (ensure your PyTorch build supports your GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is required for this run. No compatible GPU detected.")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    h5_path = '/scratch/users/kashefi/DiffPore/Data/all_data.h5'

    # Dataset & sizes
    ds = CubeDataset(h5_path)
    N = len(ds)
    print("The number of data is", N)
    n_train, n_val = int(0.8 * N), int(0.1 * N)

    # deterministic split: load or create, then wrap in Subset
    train_idx, val_idx, test_idx = load_or_create_splits(N, n_train, n_val, seed=0, prefix=Path("."))
    train_ds = Subset(ds, train_idx.tolist())
    val_ds   = Subset(ds, val_idx.tolist())
    test_ds  = Subset(ds, test_idx.tolist())

    # -------------------------
    # Compute train-only min/max for target normalization
    # -------------------------
    with h5py.File(h5_path, 'r') as f:
        perms_all = f['perms'][:]
    t_train = perms_all[train_idx]
    t_min = float(np.min(t_train))
    t_max = float(np.max(t_train))
    print(f"Train target range (physical): [{t_min:.6g}, {t_max:.6g}]")
    denom = (t_max - t_min) + 1e-12  # numerical safety

    # DataLoaders (drop_last only for training to avoid BN issue with final batch)
    num_workers = 1
    pin = True
    persist = (num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=persist, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, persistent_workers=persist, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=num_workers, pin_memory=pin, persistent_workers=persist, drop_last=False)

    # Model / loss / opt
    model = Permeability3DCNNRegressor(in_chans=1, drop_p=0.7).to(device)
    criterion = RMSELoss()  # RMSE in normalized space
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #0.001

    epochs = 1000
    train_curve, val_curve = [], []

    # measure training time
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            # normalize targets to [0,1] using TRAIN min/max
            y_norm = (yb - t_min) / denom

            pred_norm = model(Xb)               # [B] in [0,1]
            loss = criterion(pred_norm, y_norm) # RMSE in normalized space

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_idx)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_norm = (yb - t_min) / denom
                pred_norm = model(Xb)
                val_loss += criterion(pred_norm, y_norm).item() * Xb.size(0)
        val_loss /= len(val_idx)

        train_curve.append(train_loss)
        val_curve.append(val_loss)
        print(f"Epoch {epoch:03d}/{epochs}  Train RMSE(norm): {train_loss:.6f}  Val RMSE(norm): {val_loss:.6f}", flush=True)

    t1 = time.perf_counter()
    print(f"\nTotal training time: {format_hms(t1 - t0)} ({t1 - t0:.2f} seconds)")

    # Plot train/val RMSE curves (normalized)
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(range(1, epochs + 1), train_curve, label="Training")
    plt.plot(range(1, epochs + 1), val_curve, label="Validation")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.yscale('log', base=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=600)
    print(f"Saved loss plot to {Path('loss_curve.png').resolve()}")

    # Test: predict in normalized space, then map back to physical units for R^2 and plots
    model.eval()
    all_p_phys, all_t_phys = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            p_norm = model(Xb).cpu().numpy()           # [B] in [0,1]
            p_phys = p_norm * denom + t_min            # back to physical units
            all_p_phys.append(p_phys)
            all_t_phys.append(yb.numpy())              # physical targets
    all_p_phys = np.concatenate(all_p_phys)
    all_t_phys = np.concatenate(all_t_phys)

    r2 = r2_score_custom(all_t_phys, all_p_phys)
    eps = 1e-8
    rel_err = np.abs((all_p_phys - all_t_phys) / np.clip(all_t_phys, eps, None))
    print(f"\nTest R2 (physical): {r2:.4f}")
    print(f"Relative error min: {rel_err.min():.4f}, max: {rel_err.max():.4f}")

    rmse_phys = float(np.sqrt(np.mean((all_p_phys - all_t_phys) ** 2)))
    print(f"Test RMSE (physical units): {rmse_phys:.6f}")

    # R^2 scatter plot (physical units)
    plt.figure(figsize=(6, 6))
    plt.scatter(all_t_phys, all_p_phys, c='blue', s=10, label='Prediction')
    x_min, x_max = float(all_t_phys.min()), float(all_t_phys.max())
    plt.plot([x_min, x_max], [x_min, x_max], 'k--', label='y = x')
    plt.text(x_min + 0.05*(x_max - x_min),
             x_max - 0.1*(x_max - x_min),
             f"$R^2 = {r2:.4f}$", fontsize=18, ha='left', va='top')
    plt.xlabel('Ground truth (mD)', fontsize=18); plt.ylabel('Prediction (mD)', fontsize=18)
    plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout(); plt.savefig("results.png", dpi=600)
    print(f"Saved predictions vs ground-truth plot to {Path('results.png').resolve()}")

if __name__ == "__main__":
    main()
