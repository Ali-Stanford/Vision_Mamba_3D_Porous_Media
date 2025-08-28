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

# -------------------------
# Utilities
# -------------------------

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
        # x: [B, 1, 64,64,64]
        x = self.proj(x)                     # [B, C, D', H', W']
        return x

# -------------------------
# Vision-Mamba 3D Block
# -------------------------

class VMamba3DBlock(nn.Module):
    """
    Vision-Mamba-style 3D block:
      - PreNorm (GroupNorm over channels)
      - Token-dependent gating & selective SSM scan along D, H, and W axes
      - Bidirectional fusion per axis
      - Residual + MLP (conv 1x1x1)

    The selective scan uses per-token parameters B_t, C_t, ?_t (via 1x1x1 conv),
    and a learnable stable A>=0 per channel. State dimension = channels (diagonal SSM).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.GroupNorm(1, dim)  # channels-first LayerNorm-like

        # Project to 5*dim token-wise params with a 1x1x1 conv:
        # chunks: g_in | g_out | B | C | ?_raw
        self.param_proj = nn.Conv3d(dim, 5 * dim, kernel_size=1)

        # Positive diagonal A and per-channel skip D
        self.A = nn.Parameter(torch.randn(dim))   # made positive via softplus
        self.D = nn.Parameter(torch.zeros(dim))   # skip on the SSM branch

        # Second norm + MLP
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, 4 * dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(4 * dim, dim, kernel_size=1),
        )

    @staticmethod
    def _selective_scan(u_seq, B_seq, C_seq, alpha_seq, D_vec) -> torch.Tensor:
        """
        u_seq, B_seq, C_seq, alpha_seq: [Nseq, L, C], elementwise diagonal SSM per channel.
        D_vec: [C]
        Returns y_seq: [Nseq, L, C]
        """
        N, L, C = u_seq.shape
        s = u_seq.new_zeros(N, C)  # state per sequence
        y_list = []
        # forward scan
        for t in range(L):
            s = alpha_seq[:, t, :] * s + B_seq[:, t, :] * u_seq[:, t, :]
            y_t = C_seq[:, t, :] * s + D_vec * u_seq[:, t, :]
            y_list.append(y_t)
        y_fwd = torch.stack(y_list, dim=1)  # [N, L, C]

        # backward scan (on reversed time)
        s = u_seq.new_zeros(N, C)
        y_list = []
        for t in range(L - 1, -1, -1):
            s = alpha_seq[:, t, :] * s + B_seq[:, t, :] * u_seq[:, t, :]
            y_t = C_seq[:, t, :] * s + D_vec * u_seq[:, t, :]
            y_list.append(y_t)
        y_bwd = torch.stack(y_list[::-1], dim=1)  # reverse back to original order

        return 0.5 * (y_fwd + y_bwd)

    def _axis_scan(self, x: torch.Tensor,
                   g_in: torch.Tensor, g_out: torch.Tensor,
                   Bp: torch.Tensor, Cp: torch.Tensor, Delta: torch.Tensor,
                   axis: str) -> torch.Tensor:
        """
        Run selective scan along one axis.
        x, g_in, g_out, Bp, Cp, Delta: [B, C, D, H, W] (channels-first)
        axis in {'D','H','W'}
        Returns y: [B, C, D, H, W] contribution for this axis.
        """
        B, C, D, H, W = x.shape
        # Stable diagonal coefficient alpha_t = exp(-softplus(A) * ?_t)
        A_pos = F.softplus(self.A).view(1, C, 1, 1, 1)  # [1,C,1,1,1]
        alpha = torch.exp(-A_pos * Delta)               # [B,C,D,H,W]

        # Work in channels-last for easy sequencing
        def to_cl(t):  # [B,C,D,H,W] -> [B,D,H,W,C]
            return t.permute(0, 2, 3, 4, 1).contiguous()

        u_cl = to_cl(g_in * x)
        B_cl = to_cl(Bp)
        C_cl = to_cl(Cp)
        a_cl = to_cl(alpha)
        g_out_cl = to_cl(g_out)

        # Axis-specific reshape to [Nseq, L, C], scan, then restore

        if axis == 'D':
            # [B, D, H, W, C] -> Nseq=B*H*W, L=D
            u_seq = u_cl.permute(0, 2, 3, 1, 4).reshape(B * H * W, D, C)
            B_seq = B_cl.permute(0, 2, 3, 1, 4).reshape(B * H * W, D, C)
            C_seq = C_cl.permute(0, 2, 3, 1, 4).reshape(B * H * W, D, C)
            a_seq = a_cl.permute(0, 2, 3, 1, 4).reshape(B * H * W, D, C)
            y_seq = self._selective_scan(u_seq, B_seq, C_seq, a_seq, self.D)
            y_cl = y_seq.reshape(B, H, W, D, C).permute(0, 3, 1, 2, 4)  # [B,D,H,W,C]

        elif axis == 'H':
            # reorder to [B, H, D, W, C] -> Nseq=B*D*W, L=H
            u_tmp = u_cl.permute(0, 1, 3, 2, 4)  # [B, D, W, H, C] (we want H as L)
            B_tmp = B_cl.permute(0, 1, 3, 2, 4)
            C_tmp = C_cl.permute(0, 1, 3, 2, 4)
            a_tmp = a_cl.permute(0, 1, 3, 2, 4)
            # Now move H to time dimension
            u_seq = u_tmp.permute(0, 1, 2, 3, 4).reshape(B * D * W, H, C)
            B_seq = B_tmp.permute(0, 1, 2, 3, 4).reshape(B * D * W, H, C)
            C_seq = C_tmp.permute(0, 1, 2, 3, 4).reshape(B * D * W, H, C)
            a_seq = a_tmp.permute(0, 1, 2, 3, 4).reshape(B * D * W, H, C)
            y_seq = self._selective_scan(u_seq, B_seq, C_seq, a_seq, self.D)
            # Restore to [B,D,W,H,C] -> then to [B,D,H,W,C]
            y_tmp = y_seq.reshape(B, D, W, H, C)
            y_cl = y_tmp.permute(0, 1, 3, 2, 4)  # [B,D,H,W,C]

        elif axis == 'W':
            # [B, D, H, W, C] -> Nseq=B*D*H, L=W
            u_seq = u_cl.reshape(B * D * H, W, C)
            B_seq = B_cl.reshape(B * D * H, W, C)
            C_seq = C_cl.reshape(B * D * H, W, C)
            a_seq = a_cl.reshape(B * D * H, W, C)
            y_seq = self._selective_scan(u_seq, B_seq, C_seq, a_seq, self.D)
            y_cl = y_seq.reshape(B, D, H, W, C)  # already in [B,D,H,W,C]
        else:
            raise ValueError("axis must be one of {'D','H','W'}")

        # Output gate and back to channels-first
        y_cl = g_out_cl * y_cl
        y_cf = y_cl.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,D,H,W]
        return y_cf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        """
        z = self.norm1(x)

        # Token-wise parameters
        params = self.param_proj(z)                     # [B, 5C, D, H, W]
        g_in, g_out, Bp, Cp, Delta_raw = torch.chunk(params, 5, dim=1)
        g_in = torch.sigmoid(g_in)
        g_out = torch.sigmoid(g_out)
        Delta = F.softplus(Delta_raw) + 1e-4           # ensure >0

        # Axis scans
        yD = self._axis_scan(z, g_in, g_out, Bp, Cp, Delta, axis='D')
        yH = self._axis_scan(z, g_in, g_out, Bp, Cp, Delta, axis='H')
        yW = self._axis_scan(z, g_in, g_out, Bp, Cp, Delta, axis='W')
        y = (yD + yH + yW) / 3.0

        # Residual + MLP
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------
# Full Model
# -------------------------

class VMambaRegressor3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64, depth=3, patch_size=8):
        super().__init__()
        self.patch = PatchEmbed3D(in_chans, embed_dim, patch_size)
        self.blocks = nn.ModuleList([VMamba3DBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.GroupNorm(1, embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B,1,64,64,64]
        x = self.patch(x)                  # [B,C,D',H',W']
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=(2, 3, 4))          # global average pool over D',H',W'
        out = self.head(x).squeeze(-1)     # [B]
        #out = torch.sigmoid(out) #normalize between [0,1]
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


def _freeze_for_finetune(model: VMambaRegressor3D, train_last_n_blocks: int = 1):
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # 2) unfreeze last N blocks
    if train_last_n_blocks > 0:
        for blk in model.blocks[-train_last_n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    # 3) unfreeze norm + head
    for p in model.norm.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

def _count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(0)

    # -------------------------
    # Device
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ""))

    # -------------------------
    # Load model + pretrained weights
    # -------------------------
    model = VMambaRegressor3D(in_chans=1, embed_dim=64, depth=3, patch_size=8).to(device)
    ckpt_path = Path("vmamba3d_final_state.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find {ckpt_path.resolve()}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"Loaded pretrained weights from {ckpt_path.resolve()}")

    # -------------------------
    # Freeze most layers (few-shot): train only last block + norm + head
    # -------------------------
    _freeze_for_finetune(model, train_last_n_blocks=1)
    total, trainable = _count_trainable(model)
    print(f"Params: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")

    # -------------------------
    # Fixed normalization constants from your training
    # -------------------------
    t_min = 10.1419
    t_max = 199.598
    denom = (t_max - t_min) + 1e-12
    print(f"Using fixed normalization: t_min={t_min:.6f}, t_max={t_max:.6f} (denom={denom:.6f})")

    # -------------------------
    # Load FEW-SHOT dataset from .npy
    # -------------------------
    cubes_path = Path("cubes.npy")
    perms_path = Path("permeability.npy")
    if not cubes_path.exists() or not perms_path.exists():
        raise FileNotFoundError("cubes.npy and/or permeability.npy not found in current directory.")

    cubes = np.load(cubes_path)
    # accept [N,72,72,72] or [N,1,72,72,72]
    if cubes.ndim == 4:
        cubes = cubes[:, None, :, :, :]
    if cubes.ndim != 5 or cubes.shape[1] != 1:
        raise ValueError(f"Expected cubes with shape [N,72,72,72] or [N,1,72,72,72], got {cubes.shape}")
    if cubes.shape[2:] != (72, 72, 72):
        raise ValueError(f"Expected volumes of size, got {cubes.shape[2:]}")

    perms = np.load(perms_path).astype(np.float32).reshape(-1)
    if cubes.shape[0] != perms.shape[0]:
        raise ValueError(f"Sample count mismatch: cubes N={cubes.shape[0]} vs permeability N={perms.shape[0]}")

    N = cubes.shape[0]
    print(f"Few-shot dataset size: {N}")

    # Few-shot split
    FEWSHOT_TOTAL_LIMIT = int(0.9*N) #240  # total used for FT (train+val). Adjust as needed.
    subset_N = min(N, FEWSHOT_TOTAL_LIMIT)
    idx = rng.permutation(N)[:subset_N]
    split = int(0.8 * subset_N)
    train_idx_fs = idx[:split]
    val_idx_fs   = idx[split:]

    X_train = torch.from_numpy(cubes[train_idx_fs]).float()
    y_train = torch.from_numpy(perms[train_idx_fs]).float()
    X_val   = torch.from_numpy(cubes[val_idx_fs]).float()
    y_val   = torch.from_numpy(perms[val_idx_fs]).float()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds   = torch.utils.data.TensorDataset(X_val, y_val)

    # -------------------------
    # Dataloaders
    # -------------------------
    batch_size = 4  # is large; small batch
    num_workers = 1
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers>0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers>0), drop_last=False)

    # -------------------------
    # Optimizer / Loss (only trainable params)
    # -------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=0.0005, weight_decay=1e-5)
    criterion = nn.MSELoss()  # training in normalized space

    # -------------------------
    # Fine-tuning loop
    # -------------------------
    epochs = 100 #50
    best_val = float('inf')
    train_curve, val_curve = [], []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            y_norm = (yb - t_min) / denom

            pred = model(Xb)                 # normalized output
            loss = criterion(pred, y_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item() * Xb.size(0)
        tr_loss /= len(train_ds)

        # Validation
        model.eval()
        va_loss = 0.0
        preds_phys, t_phys = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_norm = (yb - t_min) / denom
                pred_norm = model(Xb)
                va_loss += criterion(pred_norm, y_norm).item() * Xb.size(0)

                # physical units for metrics
                p_phys = (pred_norm.cpu().numpy().reshape(-1) * denom) + t_min
                preds_phys.append(p_phys)
                t_phys.append(yb.cpu().numpy().reshape(-1))

        va_loss /= len(val_ds)
        train_curve.append(tr_loss)
        val_curve.append(va_loss)

        # Metrics (physical units)
        p_phys = np.concatenate(preds_phys) if preds_phys else np.array([])
        y_phys = np.concatenate(t_phys) if t_phys else np.array([])
        if y_phys.size > 0:
            r2 = r2_score_custom(y_phys, p_phys)
            rmse = float(np.sqrt(np.mean((p_phys - y_phys) ** 2)))
            print(f"Epoch {epoch:03d}/{epochs}  Train MSE(norm): {tr_loss:.6f}  "
                  f"Val MSE(norm): {va_loss:.6f}  |  Val R^2: {r2:.4f}  RMSE: {rmse:.6f}",
                  flush=True)
        else:
            print(f"Epoch {epoch:03d}/{epochs}  Train MSE(norm): {tr_loss:.6f}  Val MSE(norm): {va_loss:.6f}",
                  flush=True)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "vmamba3d_finetuned_72.pt")

    t1 = time.perf_counter()
    print(f"\nFine-tune time: {format_hms(t1 - t0)} ({t1 - t0:.2f} s)")
    print(f"Best Val MSE(norm): {best_val:.6f}  | saved -> vmamba3d_finetuned_72.pt")

    # -------------------------
    # Plot curves
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs+1), train_curve, label="Training")
    plt.plot(range(1, epochs+1), val_curve, label="Validation")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("MSE (normalized)", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("ft_loss_curve.png", dpi=600)
    print(f"Saved fine-tune loss plot to {Path('ft_loss_curve.png').resolve()}")

    # -------------------------
    # Final eval scatter on val (physical units)
    # -------------------------
    model.eval()
    preds_phys, t_phys = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device, non_blocking=True)
            pred_norm = model(Xb).cpu().numpy().reshape(-1)
            preds_phys.append(pred_norm * denom + t_min)
            t_phys.append(yb.numpy().reshape(-1))
    p_phys = np.concatenate(preds_phys) if preds_phys else np.array([])
    y_phys = np.concatenate(t_phys) if t_phys else np.array([])
    if y_phys.size > 0:
        r2 = r2_score_custom(y_phys, p_phys)
        rmse = float(np.sqrt(np.mean((p_phys - y_phys) ** 2)))
        plt.figure(figsize=(6, 6))
        plt.scatter(y_phys, p_phys, c='blue', s=10)
        x_min, x_max = float(y_phys.min()), float(y_phys.max())
        plt.plot([x_min, x_max], [x_min, x_max], 'k--')
        plt.text(x_min + 0.05*(x_max - x_min),
                 x_max - 0.1*(x_max - x_min),
                 f"$R^2 = {r2:.4f}$\nRMSE = {rmse:.4f}",
                 fontsize=14, ha='left', va='top')
        plt.xlabel('Ground truth (mD)', fontsize=14)
        plt.ylabel('Prediction (mD)', fontsize=14)
        plt.tight_layout()
        plt.savefig("ft_eval_results.png", dpi=600)
        print(f"Val R^2: {r2:.4f}  RMSE: {rmse:.6f}")
        print(f"Saved few-shot eval plot to {Path('ft_eval_results.png').resolve()}")

if __name__ == "__main__":
    main()
