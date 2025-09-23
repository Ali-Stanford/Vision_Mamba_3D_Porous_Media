import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
from typing import Tuple
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

# -------------------------
# Patch Embedding (3D)
# -------------------------

class PatchEmbed3D(nn.Module):
    """
    3D patchify stem: Conv3d with kernel=stride=patch_size.
    Returns channels-first: [B, C, D', H', W'].
    """
    def __init__(self, in_chans=1, embed_dim=64, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 1, D, H, W], typically [B,1,64,64,64]
        x = self.proj(x)                     # [B, C, D', H', W']
        return x

# -------------------------
# ViT (Transformer) blocks for 3D tokens
# -------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,   # [B, L, C]
        )
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        z = self.norm1(x)
        y, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.drop(y)
        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------
# Full ViT Regressor (3D)
# -------------------------

class ViTRegressor3D(nn.Module):
    """
    3D ViT-style regressor:
      - 3D patch embedding -> tokens [B, L, C]
      - Learned 3D absolute positional embedding (interpolated for any token grid)
      - Transformer encoder (pre-norm)
      - Global token average pooling
      - Linear head -> scalar
    """
    def __init__(self,
                 in_chans: int = 1,
                 embed_dim: int = 64,
                 depth: int = 3,
                 patch_size: int = 8,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 base_grid: Tuple[int, int, int] = (8, 8, 8)):  # default for 64^3 with p=8
        super().__init__()
        self.patch = PatchEmbed3D(in_chans, embed_dim, patch_size)
        self.embed_dim = embed_dim

        # Learned 3D pos-embed parameterized on a base grid; we interpolate at runtime
        self.base_grid = base_grid  # (D0, H0, W0)
        D0, H0, W0 = base_grid
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, D0, H0, W0))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads,
                             mlp_ratio=mlp_ratio, attn_dropout=attn_dropout,
                             dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, D, H, W]  (e.g., [B,1,64,64,64])
        """
        B = x.size(0)
        x = self.patch(x)                       # [B, C, D', H', W']
        _, C, Dp, Hp, Wp = x.shape

        # Tokens: [B, L, C]
        x_tokens = x.permute(0, 2, 3, 4, 1).contiguous().view(B, Dp * Hp * Wp, C)

        # Interpolate learned 3D pos-embed to current grid and add
        pos = F.interpolate(self.pos_embed, size=(Dp, Hp, Wp),
                            mode='trilinear', align_corners=False)  # [1, C, D', H', W']
        pos_tokens = pos.permute(0, 2, 3, 4, 1).contiguous().view(1, Dp * Hp * Wp, C)
        x_tokens = self.pos_drop(x_tokens + pos_tokens)

        # Transformer encoder
        for blk in self.blocks:
            x_tokens = blk(x_tokens)

        # Head
        x_tokens = self.norm(x_tokens)          # [B, L, C]
        pooled = x_tokens.mean(dim=1)           # global token average -> [B, C]
        out = self.head(pooled).squeeze(-1)     # [B]
        # If you want to force [0,1] range, uncomment:
        # out = torch.sigmoid(out)
        return out

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
            self.cubes = self.hf['cubes']   # shape: [N, 64,64,64]
            self.perms = self.hf['perms']   # shape: [N]

    def __len__(self):
        self._ensure_open()
        return self.perms.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        cube = self.cubes[idx]                  # (64,64,64), np.ndarray
        perm = self.perms[idx]                  # scalar
        cube = torch.from_numpy(cube).unsqueeze(0).float()  # [1,64,64,64]
        perm = torch.tensor(perm, dtype=torch.float32)
        return cube, perm

def load_or_create_splits(N, n_train, n_val, seed=0, prefix=Path(".")):
    train_p = prefix / "split_train_idx.npy"
    val_p   = prefix / "split_val_idx.npy"
    test_p  = prefix / "split_test_idx.npy"

    def valid_loaded(arr, expected_len):
        return arr is not None and arr.ndim == 1 and len(arr) == expected_len and arr.max() < N and arr.min() >= 0

    if train_p.exists() and val_p.exists() and test_p.exists():
        train_idx = np.load(train_p)
        val_idx   = np.load(val_p)
        test_idx  = np.load(test_p)
        if (len(train_idx) + len(val_idx) + len(test_idx) == N and
            len(np.intersect1d(train_idx, val_idx)) == 0 and
            len(np.intersect1d(train_idx, test_idx)) == 0 and
            len(np.intersect1d(val_idx, test_idx)) == 0 and
            valid_loaded(train_idx, n_train) and
            valid_loaded(val_idx, n_val)):
            print("Loaded existing train/val/test split indices from disk.")
            return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)
        else:
            print("Existing split files are invalid or out-of-date. Regenerating...")

    # Create fresh split
    n_test = N - n_train - n_val
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    # Save to disk
    np.save(train_p, train_idx)
    np.save(val_p, val_idx)
    np.save(test_p, test_idx)
    print(f"Saved split indices to {train_p.name}, {val_p.name}, {test_p.name}.")
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)

# -------------------------
# Training
# -------------------------

def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h5_path = '/scratch/users/kashefi/DiffPore/Data/all_data.h5'

    # Dataset & sizes
    ds = CubeDataset(h5_path)
    N = len(ds)
    print("The number of data is", N)
    n_train, n_val = int(0.8 * N), int(0.1 * N)

    # deterministic split
    train_idx, val_idx, test_idx = load_or_create_splits(N, n_train, n_val, seed=0, prefix=Path("."))
    train_ds = Subset(ds, train_idx.tolist())
    val_ds   = Subset(ds, val_idx.tolist())
    test_ds  = Subset(ds, test_idx.tolist())

    # Train target range (no leakage)
    with h5py.File(h5_path, 'r') as f:
        perms_all = f['perms'][:]
    t_train = perms_all[train_idx]
    t_min = float(np.min(t_train))
    t_max = float(np.max(t_train))
    print(f"Train target range: [{t_min:.6g}, {t_max:.6g}]")

    # DataLoaders
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              num_workers=1, pin_memory=pin, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=1, pin_memory=pin, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                             num_workers=1, pin_memory=pin, persistent_workers=True)

    # ---------------------
    # MODEL: ViT (3D)
    # ---------------------
    model = ViTRegressor3D(
        in_chans=1,
        embed_dim=64,
        depth=3,          # keep same depth as your VMamba example
        patch_size=8,
        num_heads=8,      # 64 embed_dim / 8 heads = 8 dims per head
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.0,
        base_grid=(8, 8, 8)  # for 64^3 with p=8; will be interpolated if different
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 10 #300
    train_curve, val_curve = [], []

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            y_norm = (yb - t_min) / (t_max - t_min + 1e-12)

            pred = model(Xb)
            loss = criterion(pred, y_norm)

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
                y_norm = (yb - t_min) / (t_max - t_min + 1e-12)
                val_loss += criterion(model(Xb), y_norm).item() * Xb.size(0)
        val_loss /= len(val_idx)

        train_curve.append(train_loss)
        val_curve.append(val_loss)

        print(f"Epoch {epoch:03d}/{epochs}  Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}", flush=True)

    t1 = time.perf_counter()
    print(f"\nTotal training time: {format_hms(t1 - t0)} ({t1 - t0:.2f} seconds)")

    # Plot train/val curves
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(range(1, epochs + 1), train_curve, label="Training")
    plt.plot(range(1, epochs + 1), val_curve, label="Validation")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.yscale('log', base=10)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    loss_plot_path = Path(".") / "loss_curve.png"
    plt.savefig(loss_plot_path, dpi=600)
    print(f"Saved loss plot to {loss_plot_path.resolve()}")

    # Test
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            p_norm = model(Xb).cpu().numpy()
            p = p_norm * (t_max - t_min) + t_min
            all_p.append(p)
            all_t.append(yb.numpy())
    all_p = np.concatenate(all_p)
    all_t = np.concatenate(all_t)

    r2 = r2_score_custom(all_t, all_p)
    eps = 1e-8
    rel_err = np.abs((all_p - all_t) / np.clip(all_t, eps, None))
    print(f"\nTest R2: {r2:.4f}")
    print(f"Relative error min: {rel_err.min():.4f}, max: {rel_err.max():.4f}")

    rmse_phys = float(np.sqrt(np.mean((all_p - all_t) ** 2)))
    print(f"Test RMSE (physical units): {rmse_phys:.6f}")

    # R^2 scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(all_t, all_p, c='blue', s=10, label='Prediction')
    x_min, x_max = all_t.min(), all_t.max()
    plt.plot([x_min, x_max], [x_min, x_max], 'k--', label='y = x')
    plt.text(x_min + 0.05*(x_max - x_min),
             x_max - 0.1*(x_max - x_min),
             f"$R^2 = {r2:.4f}$", fontsize=18, ha='left', va='top')
    plt.xlabel('Ground truth (mD)', fontsize=18)
    plt.ylabel('Prediction (mD)', fontsize=18)
    plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.savefig("results.png", dpi=600)
    print(f"Saved predictions vs ground-truth plot to {Path('results.png').resolve()}")

    final_state_path = Path(".") / "vit3d_final_state.pt"
    torch.save(model.state_dict(), final_state_path)
    print(f"Saved final model state_dict to {final_state_path.resolve()}")

if __name__ == "__main__":
    main()
