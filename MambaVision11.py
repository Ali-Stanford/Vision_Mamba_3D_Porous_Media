import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader, random_split
import h5py

# custom R2 function
def r2_score_custom(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot

class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register_parameter('log_dt', nn.Parameter(log_dt) if lr is None else nn.Parameter(log_dt))
        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register_parameter('log_A_real', nn.Parameter(log_A_real))
        self.register_buffer('A_imag', A_imag)

    def forward(self, L):
        dt = torch.exp(self.log_dt)                 # (H)
        C = torch.view_as_complex(self.C)           # (H, N/2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)
        dtA = A * dt.unsqueeze(-1)                  # (H, N/2)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N/2, L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum('hn,hnl->hl', C, torch.exp(K)).real  # (H, L)
        return K

class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=False, **kernel_args):
        super().__init__()
        self.h = d_model
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=d_state, **kernel_args)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=1),
        )
        self.transposed = transposed

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)  # [B, H, L]
        L = u.size(-1)
        k = self.kernel(L=L)       # [H, L]
        k_f = torch.fft.rfft(k, n=2*L)        # [H, L']
        u_f = torch.fft.rfft(u, n=2*L)        # [B, H, L']
        y = torch.fft.irfft(u_f * k_f, n=2*L)[..., :L]  # [B, H, L]
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)            # [B, H, L]
        if not self.transposed:
            y = y.transpose(-1, -2)          # [B, L, H]
        return y, None

# 3D patch embedding
class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                      # [B, embed_dim, D', H', W']
        B, C, D, H, W = x.shape
        x = x.view(B, C, D*H*W)               # [B, C, seq_len]
        return x.permute(0, 2, 1)             # [B, seq-len, C]

# Mamba block using S4D
class MambaBlock(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.s4d   = S4D(d_model=dim, d_state=dim, dropout=0.1, transposed=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )

    def forward(self, x):
        # x: [B, seq_len, dim]
        # Pre-norm
        res = self.norm1(x)

        # Forward S4D
        y_fwd, _ = self.s4d(res)  # [B, seq_len, dim]

        # Backward S4D: reverse sequence, apply S4D, then reverse output
        res_rev = torch.flip(res, dims=[1])
        y_rev, _ = self.s4d(res_rev)
        y_rev = torch.flip(y_rev, dims=[1])

        # Combine forward and backward passes
        y = (y_fwd + y_rev) * 0.5

        # Residual connection
        x = x + y

        # Feed-forward with second residual
        res2 = self.norm2(x)
        x = x + self.ff(res2)

        return x

# full 3D regressor
class MambaRegressor3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=32, depth=2, patch_size=16):
        super().__init__()
        self.patch = PatchEmbed3D(in_chans, embed_dim, patch_size)
        seq_len = (64 // patch_size) ** 3
        self.blocks = nn.ModuleList([MambaBlock(embed_dim, seq_len) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed_dim)
        #self.head   = nn.Sequential(nn.Linear(embed_dim,1), nn.Sigmoid())
        self.head   = nn.Linear(embed_dim,1)
    def forward(self, x):
        x = self.patch(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x).squeeze(-1)

class CubeDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.hf = None

    def _ensure_open(self):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')
            self.cubes = self.hf['cubes']
            self.perms = self.hf['perms']

    def __len__(self):
        self._ensure_open()
        return self.perms.shape[0]

    def __getitem__(self, idx):
        self._ensure_open()
        cube = self.cubes[idx]             # (D, H, W)
        perm = self.perms[idx]             # ()
        cube = torch.from_numpy(cube)      # CPU tensor
        cube = cube.unsqueeze(0).float()   # add channel dim
        perm = torch.tensor(perm).float()
        return cube, perm

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h5_path = '/scratch/users/kashefi/DiffPore/Data/all_data.h5'


    ds = CubeDataset(h5_path)
    N = len(ds)
    print("The number of data is ", N)
    n_train, n_val = int(0.8 * N), int(0.1 * N)
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0)
    )

    # compute t_min, t_max by streaming through train_ds
    t_min, t_max = float('inf'), float('-inf')
    stats_loader = DataLoader(train_ds, batch_size=256,
                              num_workers=4, pin_memory=True)
    for _, y in stats_loader:
        t_min = min(t_min, y.min().item())
        t_max = max(t_max, y.max().item())
    
    # now standard DataLoaders for train/val/test
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )             

    model     = MambaRegressor3D().to(device)
    #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 270 #350 #400
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            y_norm = (yb - t_min)/(t_max - t_min)
            pred = model(Xb)
            loss = criterion(pred, y_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*Xb.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                y_norm = (yb - t_min)/(t_max - t_min)
                val_loss += criterion(model(Xb), y_norm).item()*Xb.size(0)
        val_loss /= n_val

        print(f"Epoch {epoch}/{epochs}  Train MSE: {train_loss:.4f},  Val MSE: {val_loss:.4f}",flush=True)

    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            p_norm = model(Xb).cpu().numpy()
            p = p_norm*(t_max - t_min) + t_min
            all_p.append(p)
            all_t.append(yb.numpy())
    all_p = np.concatenate(all_p)
    all_t = np.concatenate(all_t)

    r2 = r2_score_custom(all_t, all_p)
    rel_err = np.abs((all_p - all_t)/all_t)
    print(f"\nTest R2: {r2:.4f}")
    print(f"Relative error min: {rel_err.min():.4f}, max: {rel_err.max():.4f}")

if __name__ == "__main__":
    main()
