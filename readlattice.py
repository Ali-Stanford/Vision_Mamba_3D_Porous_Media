import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def has_path(lattice):
    n = lattice.shape[0]
    visited = np.zeros_like(lattice, dtype=bool)
    # enqueue all pore cells (0) on the left edge
    queue = deque([(i, 0) for i in range(n) if lattice[i, 0] == 0])
    for i, _ in queue:
        visited[i, 0] = True

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        x, y = queue.popleft()
        # reached right edge?
        if y == n - 1:
            return True
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < n and 0 <= ny < n and
                not visited[nx, ny] and lattice[nx, ny] == 0):
                visited[nx, ny] = True
                queue.append((nx, ny))
    return False

#######
def compute_effective_permeability_x(grid, k0=1.0, mu=1.0, dx=0.003):
    grid = np.transpose(grid)
    nx, ny = grid.shape
    pore = (grid == 0)

    # --- 1) find the connected pore network reachable from the left boundary ---
    visited = set()
    queue = deque( (0, j) for j in range(ny) if pore[0, j] )
    while queue:
        i, j = queue.popleft()
        if (i,j) in visited:
            continue
        visited.add((i,j))
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < nx and 0 <= nj < ny and pore[ni, nj] and (ni,nj) not in visited:
                queue.append((ni,nj))

    # mask of the “active” pore network
    active = np.zeros_like(grid, bool)
    for (i,j) in visited:
        active[i,j] = True


    # ensure we have boundary connectivity
    left_active  = [ (0,j)      for j in range(ny) if active[0,j] ]
    right_active = [ (nx-1,j)   for j in range(ny) if active[nx-1,j] ]
    if not left_active or not right_active:
        # no percolating cluster → zero permeability
        return 0.0

    # --- 2) re‑index only the active pores ---
    idx = -np.ones_like(grid, int)
    active_list = list(visited)
    for new_id, (i,j) in enumerate(active_list):
        idx[i,j] = new_id
    N = len(active_list)

    # --- 3) build the sparse matrix A and RHS b ---
    rows, cols, data = [], [], []
    b = np.zeros(N)
    g = k0/(mu*dx)

    for p_id, (i,j) in enumerate(active_list):
        diag = 0.0

        # neighbors in 4 directions
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < nx and 0 <= nj < ny and active[ni,nj]:
                q_id = idx[ni,nj]
                rows.append(p_id); cols.append(q_id); data.append(-g)
                diag += g
            else:
                # handle Dirichlet at left/right
                if ni < 0:
                    diag += g
                    b[p_id] += g * 1.0    # P_left = 1
                elif ni >= nx:
                    diag += g            # P_right=0 → contributes diag only
                # top/bottom outside active but i in [0,nx), treat as no-flux

        rows.append(p_id); cols.append(p_id); data.append(diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N,N))
    #print(A)

    # --- 4) solve for pressures ---
    P = spla.spsolve(A, b)

    # --- 5) compute total flow Q across the right boundary ---
    Q = 0.0
    for j in range(ny):
        if active[nx-1, j]:
            p_id = idx[nx-1, j]
            Q += g * (P[p_id] - 0.0)

    # --- 6) extract k_eff via Darcy’s law ---
    A_cross = ny * dx    # unit depth
    L       = nx * dx
    ΔP      = 1.0
    k_eff   = (Q / A_cross) * (mu * L / ΔP)

    return k_eff
#######

def main():
   
    binary_string = """
    01010100000100101010001010010011
    00101110101101100100110101101000
    01010110101100011100011100000101
    00111110100110100110001011100011
    01000110010100110000010100010100
    01001010110101011110100001000110
    00000010101001100010100011110100
    00000010100010000000001000010111
    01000011101100100000100010100100
    00000110110110010000101100100010
    01100110010000001000000100010100
    11110100001110110000000111100100
    01001001110000000110100100010101
    10100010000000010000111111101100
    00110110010000001101000010001001
    10000101000100100101010100111101
    00100100000100111100100000010100
    00001000001000001001101101001101
    00001101010001000001001000000010
    00001010100100011100001010000010
    00000001101110101101000010010111
    01000100000100001000100010000100
    00001000000000101100010010101001
    00000010100101010000110000001001
    01000010001110111110100101010101
    10000110001100101100010000110100
    01000110010000100011110010110100
    10101100000010101001010001100011
    00010110010000100010000100001010
    00001110000100001110001001000001
    00001100101001001110110100001000
    01000000100000010100100001001011
    """
    
    lines = binary_string.strip().split('\n')
    lattice = np.array([[int(char) for char in line.strip()] for line in lines])
            
    porosity = np.mean(lattice == 0)
       
    print(porosity)
     
    if has_path(lattice):
        print(f"Found lattice with porosity {porosity:.3f}")
        k_eff = compute_effective_permeability_x(lattice)
        print("Effective permeability k_eff =", k_eff)
                
    # Plot and save
    plt.imshow(lattice, cmap='gray_r', origin='upper')
    plt.axis('off')
    plt.title(f'Percolating {lattice.shape[0]} by {lattice.shape[1]} Lattice (porosity={porosity:.3f})')
    plt.savefig('lattice.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
