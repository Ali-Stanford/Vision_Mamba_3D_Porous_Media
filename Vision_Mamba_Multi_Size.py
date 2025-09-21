import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Utilities
# =========================

def r2_score_custom(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def format_hms(seconds: float) -> str:
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = seconds - (h*3600 + m*60)
    return f"{h}h {m}m {s:.1f}s"

# =========================
# Patch Embedding (3D)
# =========================

class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
    def forward(self, x):  # x: [B,1,D,H,W], D/H/W % patch_size == 0
        return self.proj(x)

# =========================
# VMamba 3D Block
# =========================

class VMamba3DBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.GroupNorm(1, dim)
        self.param_proj = nn.Conv3d(dim, 5 * dim, kernel_size=1)
        self.A = nn.Parameter(torch.randn(dim))
        self.D = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, 4 * dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(4 * dim, dim, kernel_size=1),
        )

    @staticmethod
    def _selective_scan(u_seq, B_seq, C_seq, alpha_seq, D_vec):
        N, L, C = u_seq.shape
        s = u_seq.new_zeros(N, C)
        y_list = []
        for t in range(L):
            s = alpha_seq[:, t, :] * s + B_seq[:, t, :] * u_seq[:, t, :]
            y_t = C_seq[:, t, :] * s + D_vec * u_seq[:, t, :]
            y_list.append(y_t)
        y_fwd = torch.stack(y_list, dim=1)

        s = u_seq.new_zeros(N, C)
        y_list = []
        for t in range(L - 1, -1, -1):
            s = alpha_seq[:, t, :] * s + B_seq[:, t, :] * u_seq[:, t, :]
            y_t = C_seq[:, t, :] * s + D_vec * u_seq[:, t, :]
            y_list.append(y_t)
        y_bwd = torch.stack(y_list[::-1], dim=1)
        return 0.5 * (y_fwd + y_bwd)

    def _axis_scan(self, x, g_in, g_out, Bp, Cp, Delta, axis: str):
        B, C, D, H, W = x.shape
        A_pos = F.softplus(self.A).view(1, C, 1, 1, 1)
        alpha = torch.exp(-A_pos * Delta)

        def to_cl(t):  # [B,C,D,H,W] -> [B,D,H,W,C]
            return t.permute(0, 2, 3, 4, 1).contiguous()

        u_cl, B_cl, C_cl, a_cl, g_out_cl = to_cl(g_in * x), to_cl(Bp), to_cl(Cp), to_cl(alpha), to_cl(g_out)

        if axis == 'D':
            u_seq = u_cl.permute(0,2,3,1,4).reshape(B*H*W, D, C)
            B_seq = B_cl.permute(0,2,3,1,4).reshape(B*H*W, D, C)
            C_seq = C_cl.permute(0,2,3,1,4).reshape(B*H*W, D, C)
            a_seq = a_cl.permute(0,2,3,1,4).reshape(B*H*W, D, C)
            y_seq = self._selective_scan(u_seq, B_seq, C_seq, a_seq, self.D)
            y_cl  = y_seq.reshape(B, H, W, D, C).permute(0,3,1,2,4)
        elif axis == 'H':
            u_tmp = u_cl.permute(0,1,3,2,4)  # [B,D,W,H,C]
            B_tmp = B_cl.permute(0,1,3,2,4)
            C_tmp = C_cl.permute(0,1,3,2,4)
            a_tmp = a_cl.permute(0,1,3,2,4)
            u_seq = u_tmp.reshape(B*D*W, H, C)
            B_seq = B_tmp.reshape(B*D*W, H, C)
            C_seq = C_tmp.reshape(B*D*W, H, C)
            a_seq = a_tmp.reshape(B*D*W, H, C)
            y_seq = self._selective_scan(u_seq, B_seq, C_seq, a_seq, self.D)
            y_tmp = y_seq.reshape(B, D, W, H, C)
            y_cl  = y_tmp.permute(0,1,3,2,4)
        elif axis == 'W':
            u_seq = u_cl.reshape(B*D*H, W, C)
            B_seq = B_cl.reshape(B*D*H, W, C)
            C_seq = C_cl.reshape(B*D*H, W, C)
            a_seq = a_cl.reshape(B*D*H, W, C)
            y_seq = self._selective_scan(u_seq, B_seq, C_seq, a_seq, self.D)
            y_cl  = y_seq.reshape(B, D, H, W, C)
        else:
            raise ValueError("axis must be one of {'D','H','W'}")

        y_cf = (g_out_cl * y_cl).permute(0,4,1,2,3).contiguous()
        return y_cf

    def forward(self, x):
        z = self.norm1(x)
        params = self.param_proj(z)  # [B,5C,D,H,W]
        g_in, g_out, Bp, Cp, Delta_raw = torch.chunk(params, 5, dim=1)
        g_in = torch.sigmoid(g_in); g_out = torch.sigmoid(g_out)
        Delta = F.softplus(Delta_raw) + 1e-4

        y = (self._axis_scan(z, g_in, g_out, Bp, Cp, Delta, 'D') +
             self._axis_scan(z, g_in, g_out, Bp, Cp, Delta, 'H') +
             self._axis_scan(z, g_in, g_out, Bp, Cp, Delta, 'W')) / 3.0
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

# =========================
# Full Model
# =========================

class VMambaRegressor3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64, depth=3, patch_size=8):
        super().__init__()
        self.patch = PatchEmbed3D(in_chans, embed_dim, patch_size)
        self.blocks = nn.ModuleList([VMamba3DBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.GroupNorm(1, embed_dim)
        self.head = nn.Linear(embed_dim, 1)
    def forward(self, x):
        x = self.patch(x)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=(2,3,4))
        return self.head(x).squeeze(-1)

# =========================
# Datasets from .npy
# =========================

class CubeDatasetNPY(Dataset):
    """
    Loads cubes_{size}.npy (N,D,H,W) and permeability_{size}.npy (N,)
    Uses numpy memmap for low RAM.
    """
    def __init__(self, base_dir: str, size: int):
        self.size = int(size)
        self.cubes = np.load(os.path.join(base_dir, f"cubes_{self.size}.npy"), mmap_mode='r')
        self.perms = np.load(os.path.join(base_dir, f"permeability_{self.size}.npy"), mmap_mode='r')
        assert self.cubes.ndim == 4 and self.cubes.shape[1:] == (self.size, self.size, self.size), \
            f"cubes_{self.size}.npy must be [N,{self.size},{self.size},{self.size}]"
        assert self.perms.shape[0] == self.cubes.shape[0], "permeability length must match cubes N"
    def __len__(self):
        return self.perms.shape[0]
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cube = self.cubes[idx]  # (D,H,W)
        perm = self.perms[idx]
        cube = torch.from_numpy(cube).unsqueeze(0).float()  # [1,D,H,W]
        perm = torch.tensor(perm, dtype=torch.float32)
        return cube, perm
    def get_shape(self):
        return tuple(self.cubes.shape[1:])  # (D,H,W)

# =========================
# Round-robin mixer
# =========================

def round_robin_batches(*loaders):
    iters = [iter(l) for l in loaders]
    alive = [True] * len(iters)
    while any(alive):
        for i, it_ in enumerate(iters):
            if not alive[i]: continue
            try:
                yield i, next(it_)
            except StopIteration:
                alive[i] = False

# =========================
# Deterministic in-code splits (0.8 / 0.1 / 0.1)
# =========================

def make_splits_in_code(N: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    train = perm[:n_train]
    val   = perm[n_train:n_train + n_val]
    test  = perm[n_train + n_val:]
    return {"train": train.astype(int), "val": val.astype(int), "test": test.astype(int)}

# =========================
# Training script
# =========================

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Update this path if needed ----
    #BASE_DIR = "/scratch/users/kashefi/DiffPore/Data"
    BASE_DIR = "/scratch/users/kashefi/DiffPore/Train/MambaVision/p21_mixed_72_80_88"
    SIZES = [72, 80, 88]
    patch_size = 8  # divides 72,80,88
    # Tweak per-size batches if memory-bound:
    BATCH_SIZES = {72: 8, 80: 6, 88: 4}
    num_workers = 1
    epochs = 300
    lr = 1e-3
    wd = 1e-4
    seed = 0  # controls the in-code split

    # Datasets, in-code splits, loaders per size
    datasets = {}
    splits = {}
    loaders = {}
    pin = (device.type == 'cuda')

    for sz in SIZES:
        ds = CubeDatasetNPY(BASE_DIR, sz)
        N = len(ds)
        D,H,W = ds.get_shape()
        assert D % patch_size == 0 and H % patch_size == 0 and W % patch_size == 0, \
            f"Size {D}x{H}x{W} must be divisible by patch_size={patch_size}"

        sp = make_splits_in_code(N, seed=seed + sz)  # different seed per size for variety
        splits[sz] = sp
        datasets[sz] = ds

        loaders[sz] = {
            "train": DataLoader(Subset(ds, sp["train"].tolist()), batch_size=BATCH_SIZES[sz], shuffle=True,
                                num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers>0)),
            "val":   DataLoader(Subset(ds, sp["val"].tolist()), batch_size=BATCH_SIZES[sz], shuffle=False,
                                num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers>0)),
            "test":  DataLoader(Subset(ds, sp["test"].tolist()), batch_size=BATCH_SIZES[sz], shuffle=False,
                                num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers>0)),
        }
        print(f"[{sz}] N={N}  Train={len(sp['train'])}  Val={len(sp['val'])}  Test={len(sp['test'])}")

    # Global normalization from ALL training targets (no leakage)
    train_targets = []
    for sz in SIZES:
        perms = datasets[sz].perms  # memmap
        train_targets.append(perms[splits[sz]["train"]])
    train_targets = np.concatenate(train_targets)
    t_min, t_max = float(np.min(train_targets)), float(np.max(train_targets))
    print(f"Global train target range: [{t_min:.6g}, {t_max:.6g}]")

    # Model / loss / opt
    model = VMambaRegressor3D(in_chans=1, embed_dim=64, depth=3, patch_size=patch_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _norm_targets(y): return (y - t_min) / (t_max - t_min + 1e-12)

    # ---- Training loop (round-robin across sizes) ----
    train_curve, val_curve = [], []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss, total_train_n = 0.0, 0

        rr = round_robin_batches(loaders[72]["train"], loaders[80]["train"], loaders[88]["train"])
        for loader_id, (Xb, yb) in rr:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            y_norm = _norm_targets(yb)

            pred = model(Xb)
            loss = criterion(pred, y_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = Xb.size(0)
            total_train_loss += loss.item() * bs
            total_train_n += bs

        train_loss = total_train_loss / max(total_train_n, 1)

        # Validation (size-weighted average)
        model.eval()
        total_val_loss, total_val_n = 0.0, 0
        with torch.no_grad():
            for sz in SIZES:
                for Xb, yb in loaders[sz]["val"]:
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    y_norm = _norm_targets(yb)
                    l = criterion(model(Xb), y_norm).item()
                    total_val_loss += l * Xb.size(0)
                    total_val_n += Xb.size(0)
        val_loss = total_val_loss / max(total_val_n, 1)

        train_curve.append(train_loss); val_curve.append(val_loss)
        print(f"Epoch {epoch:03d}/{epochs}  Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}", flush=True)

    t1 = time.perf_counter()
    print(f"\nTotal training time: {format_hms(t1 - t0)} ({t1 - t0:.2f} seconds)")

    # ---- Plot curves ----
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(range(1, epochs + 1), train_curve, label="Training")
    plt.plot(range(1, epochs + 1), val_curve, label="Validation")
    plt.xlabel("Epoch", fontsize=18); plt.ylabel("Loss", fontsize=18)
    plt.yscale('log', base=10); plt.legend(fontsize=14); plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout(); plt.savefig("loss_curve_multi_size_incode_splits.png", dpi=600)
    print(f"Saved loss plot to {Path('loss_curve_multi_size_incode_splits.png').resolve()}")

    # ---- Test (per-size and overall) ----
    def eval_loader(dl):
        all_p, all_t = [], []
        with torch.no_grad():
            for Xb, yb in dl:
                Xb = Xb.to(device, non_blocking=True)
                p_norm = model(Xb).cpu().numpy()
                p = p_norm * (t_max - t_min) + t_min
                all_p.append(p); all_t.append(yb.numpy())
        if not all_p:
            return dict(r2=np.nan, rmse=np.nan, rel_min=np.nan, rel_max=np.nan, all_p=np.array([]), all_t=np.array([]))
        all_p = np.concatenate(all_p); all_t = np.concatenate(all_t)
        r2 = r2_score_custom(all_t, all_p)
        eps = 1e-8
        rel = np.abs((all_p - all_t) / np.clip(all_t, eps, None))
        rmse = float(np.sqrt(np.mean((all_p - all_t) ** 2)))
        return dict(r2=r2, rmse=rmse, rel_min=float(rel.min()), rel_max=float(rel.max()),
                    all_p=all_p, all_t=all_t)

    results = {}
    for sz in SIZES:
        results[sz] = eval_loader(loaders[sz]["test"])
        print(f"[Test {sz}] R2={results[sz]['r2']:.4f}  RMSE={results[sz]['rmse']:.6f}  "
              f"RelErr[min,max]=({results[sz]['rel_min']:.4f}, {results[sz]['rel_max']:.4f})")

    all_p_all = np.concatenate([results[sz]["all_p"] for sz in SIZES])
    all_t_all = np.concatenate([results[sz]["all_t"] for sz in SIZES])
    r2_all = r2_score_custom(all_t_all, all_p_all)
    eps = 1e-8
    rel_all = np.abs((all_p_all - all_t_all) / np.clip(all_t_all, eps, None))
    rmse_all = float(np.sqrt(np.mean((all_p_all - all_t_all) ** 2)))
    print(f"\n[Test Overall] R2={r2_all:.4f}  RMSE={rmse_all:.6f}  "
          f"RelErr[min,max]=({rel_all.min():.4f}, {rel_all.max():.4f})")

    plt.figure(figsize=(6,6))
    plt.scatter(all_t_all, all_p_all, s=10)
    x_min, x_max = all_t_all.min(), all_t_all.max()
    plt.plot([x_min, x_max], [x_min, x_max], 'k--')
    plt.text(x_min + 0.05*(x_max - x_min), x_max - 0.1*(x_max - x_min),
             f"$R^2 = {r2_all:.4f}$", fontsize=18, ha='left', va='top')
    plt.xlabel('Ground truth (mD)', fontsize=18); plt.ylabel('Prediction (mD)', fontsize=18)
    plt.tick_params(axis='both', labelsize=10); plt.tight_layout()
    plt.savefig("results_multi_size_incode_splits.png", dpi=600)
    print(f"Saved predictions vs ground-truth plot to {Path('results_multi_size_incode_splits.png').resolve()}")

    torch.save(model.state_dict(), "vmamba3d_final_state_incode_splits.pt")
    print(f"Saved final model state_dict to {Path('vmamba3d_final_state_incode_splits.pt').resolve()}")

if __name__ == "__main__":
    main()
