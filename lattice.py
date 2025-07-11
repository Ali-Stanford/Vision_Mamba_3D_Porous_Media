import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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

def main():
    n = 32
    min_porosity = 0.2
    max_porosity = 0.5

    while True:
        lattice = np.random.randint(0, 2, (n, n))
        porosity = np.mean(lattice == 0)
        # check porosity constraint first
        if not (min_porosity <= porosity <= max_porosity):
            continue
        # then check percolation
        if has_path(lattice):
            print(f"Found lattice with porosity {porosity:.3f}")
            break

    # Save lattice as 16 lines of 16 numbers
    np.savetxt('lattice.txt', lattice, fmt='%d')

    # Plot and save
    plt.imshow(lattice, cmap='gray_r', origin='upper')
    plt.axis('off')
    plt.title(f'Percolating 16×16 Lattice (porosity={porosity:.3f})')
    plt.savefig('lattice.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
