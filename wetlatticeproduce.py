import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def generate_binary_gaussian_field(size, sigma, correlation_length, threshold):
    kx = np.fft.fftfreq(size).reshape(-1, 1)
    ky = np.fft.fftfreq(size).reshape(1, -1)
    k_squared = kx**2 + ky**2

    power_spectrum = np.exp(-2 * (np.pi**2) * k_squared * (correlation_length**2))
    noise = np.fft.fft2(np.random.randn(size, size))
    filtered_noise = noise * np.sqrt(power_spectrum)
    field = np.fft.ifft2(filtered_noise).real

    field -= field.min()
    field /= field.max()
    return (field >= threshold).astype(int)

def has_path(lattice):
    n = lattice.shape[0]
    visited = np.zeros_like(lattice, dtype=bool)
    queue = deque((i, 0) for i in range(n) if lattice[i, 0] == 0)
    for i, j in queue:
        visited[i, j] = True

    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    while queue:
        x, y = queue.popleft()
        if y == n - 1:
            return True
        for dx, dy in moves:
            nx_, ny_ = x+dx, y+dy
            if (0 <= nx_ < n and 0 <= ny_ < n
                and not visited[nx_, ny_]
                and lattice[nx_, ny_] == 0):
                visited[nx_, ny_] = True
                queue.append((nx_, ny_))
    return False

def compute_effective_permeability_x(grid, k0=1.0, mu=1.0, dx=0.003):
    grid = grid.T
    nx, ny = grid.shape
    pore = (grid == 0)

    visited = set()
    queue = deque((0, j) for j in range(ny) if pore[0, j])
    while queue:
        i, j = queue.popleft()
        if (i,j) in visited:
            continue
        visited.add((i,j))
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if (0 <= ni < nx and 0 <= nj < ny
                and pore[ni, nj]
                and (ni,nj) not in visited):
                queue.append((ni, nj))

    active = np.zeros_like(grid, bool)
    for (i, j) in visited:
        active[i, j] = True

    left_conn  = any(active[0, j]   for j in range(ny))
    right_conn = any(active[nx-1, j] for j in range(ny))
    if not (left_conn and right_conn):
        return 0.0

    idx = -np.ones_like(grid, int)
    for new_id, (i, j) in enumerate(visited):
        idx[i, j] = new_id
    N = len(visited)

    rows, cols, data = [], [], []
    b = np.zeros(N)
    g = k0/(mu*dx)

    for p_id, (i, j) in enumerate(visited):
        diag = 0.0
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < nx and 0 <= nj < ny and active[ni, nj]:
                q_id = idx[ni, nj]
                rows.append(p_id); cols.append(q_id); data.append(-g)
                diag += g
            else:
                if ni < 0:
                    diag += g
                    b[p_id] += g * 1.0
                elif ni >= nx:
                    diag += g
        rows.append(p_id); cols.append(p_id); data.append(diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    P = spla.spsolve(A, b)

    Q = 0.0
    for j in range(ny):
        if active[nx-1, j]:
            p_id = idx[nx-1, j]
            Q += g * (P[p_id] - 0.0)

    A_cross = ny * dx
    L       = nx * dx
    ΔP      = 1.0
    return (Q / A_cross) * (mu * L / ΔP)

def bfs_mark(lattice, start_edge):
    """
    BFS from either 'left' or 'right' edge, returns boolean mask.
    """
    n = lattice.shape[0]
    visited = np.zeros_like(lattice, dtype=bool)
    if start_edge == 'left':
        queue = deque((i, 0) for i in range(n) if lattice[i, 0] == 0)
    else:  # 'right'
        queue = deque((i, n-1) for i in range(n) if lattice[i, n-1] == 0)

    for i, j in queue:
        visited[i, j] = True

    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in moves:
            nx_, ny_ = x+dx, y+dy
            if (0 <= nx_ < n and 0 <= ny_ < n
                and not visited[nx_, ny_]
                and lattice[nx_, ny_] == 0):
                visited[nx_, ny_] = True
                queue.append((nx_, ny_))
    return visited

def main():
    n = 32
    min_porosity, max_porosity = 0.2, 0.5
    output_lines = []

    for item in range(10):
        while True:
            lattice = np.random.choice([0,1], size=(n,n), p=[0.5,0.5])
            porosity = (lattice == 0).mean()
            if not (min_porosity <= porosity <= max_porosity):
                continue
            if has_path(lattice):
                print(f"Found lattice with porosity {porosity:.3f}")
                k_eff = compute_effective_permeability_x(lattice)
                print("Effective permeability k_eff =", k_eff)
                break

        # mark left- and right-reachable pores
        left_vis  = bfs_mark(lattice, 'left')
        right_vis = bfs_mark(lattice, 'right')
        wet_lattice = (left_vis & right_vis).astype(int)

        eff_porosity = wet_lattice.sum() / wet_lattice.size
        print(f"Effective porosity (wet cluster) = {eff_porosity:.3f}")

        np.savetxt(f"wet_lattice{item}.txt", wet_lattice, fmt='%d')

        output_lines.append(f"Example {item+1}:")
        #output_lines.append(f"Porosity: {porosity:.2f}")
        output_lines.append(f"Porosity: {eff_porosity:.2f}")
        output_lines.append(f"Permeability: {k_eff:.4f}")
        output_lines.append("Structure:")
        #for row in lattice:
        for row in wet_lattice:
            output_lines.append(''.join(map(str, row)))
        output_lines.append("")

    with open("porous_examples.txt", "w") as f:
        f.write("\n".join(output_lines))

    # plot original lattice (pores white, grains black)
    plt.imshow(lattice, cmap='gray_r', origin='upper')
    plt.axis('off')
    plt.title(f'Original Lattice (porosity={porosity:.3f})')
    plt.savefig('lattice.png', dpi=300, bbox_inches='tight')
    plt.show()

    # plot wet cluster (wet pores white, everything else black)
    plt.figure()
    plt.imshow(wet_lattice, cmap='gray', origin='upper')
    plt.axis('off')
    plt.title(f'Wet Pore Cluster (eff. porosity={eff_porosity:.3f})')
    plt.savefig('wet_lattice.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
