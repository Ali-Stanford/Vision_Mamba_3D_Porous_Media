# Vision-Mamba-style 3D permeability regressor (MSE)
# - Reads HDF5: datasets "cubes" (64x64x64) and "perms" (scalar)
# - Patchifies 3D volumes, then stacks VMamba3DBlocks
# - Selective scans along D, H, W with bidirectional fusion
# - Outputs scalar permeability via MSE on train-time min-max normalized targets

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from typing import Tuple

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

# -------------------------
# Training
# -------------------------

def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h5_path = '/scratch/users/kashefi/DiffPore/Data/all_data.h5'

    # Dataset & split
    ds = CubeDataset(h5_path)
    N = len(ds)
    print("The number of data is", N)
    n_train, n_val = int(0.8 * N), int(0.1 * N)
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0)
    )

    # Get target min/max directly from HDF5 once (faster than streaming cubes)
    with h5py.File(h5_path, 'r') as f:
        perms_all = f['perms'][:]  # small 1D array
    # Use only train split indices for min/max normalization
    train_indices = train_ds.indices if hasattr(train_ds, 'indices') else range(n_train)
    t_train = perms_all[train_indices]
    t_min = float(np.min(t_train))
    t_max = float(np.max(t_train))
    print(f"Train target range: [{t_min:.6g}, {t_max:.6g}]")

    # DataLoaders
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              num_workers=4, pin_memory=pin, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=4, pin_memory=pin, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                             num_workers=4, pin_memory=pin, persistent_workers=True)

    # Model / loss / opt
    model = VMambaRegressor3D(in_chans=1, embed_dim=64, depth=3, patch_size=8).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 100
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
        train_loss /= n_train

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_norm = (yb - t_min) / (t_max - t_min + 1e-12)
                val_loss += criterion(model(Xb), y_norm).item() * Xb.size(0)
        val_loss /= n_val

        print(f"Epoch {epoch:03d}/{epochs}  Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}", flush=True)

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

if __name__ == "__main__":
    main()
