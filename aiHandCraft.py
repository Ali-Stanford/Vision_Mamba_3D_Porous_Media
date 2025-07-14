import numpy as np
import random

def generate_structure(size=32, porosity_min=0.25, porosity_max=0.5):
    # Carve a guaranteed left-to-right pore path
    def carve_path():
        r = random.randrange(size)
        c = 0
        path = [(r, c)]
        while c < size - 1:
            moves = []
            if r > 0:
                moves.append((r - 1, c))
            if r < size - 1:
                moves.append((r + 1, c))
            moves.append((r, c + 1))
            r, c = random.choice(moves)
            path.append((r, c))
        return set(path)

    path = carve_path()
    total_cells = size * size
    # Determine target number of pores
    porosity = random.uniform(porosity_min, porosity_max)
    target_zeros = int(round(porosity * total_cells))
    # Initialize grid with solids
    grid = np.ones((size, size), dtype=int)
    # Carve path cells as pores
    for (i, j) in path:
        grid[i, j] = 0
    # Allocate remaining pores
    remaining = target_zeros - len(path)
    candidates = [(i, j) for i in range(size) for j in range(size) if (i, j) not in path]
    chosen = random.sample(candidates, max(0, remaining))
    for (i, j) in chosen:
        grid[i, j] = 0
    # Enforce no all-zero or all-one rows/columns
    def fix_line(line_indices, is_row=True):
        arr = grid[line_indices] if is_row else grid[:, line_indices]
        zeros = np.sum(arr == 0)
        if zeros == 0:
            # flip one solid to pore
            idx = random.choice(range(size))
            i, j = (line_indices, idx) if is_row else (idx, line_indices)
            if (i, j) not in path:
                grid[i, j] = 0
        elif zeros == size:
            # flip one pore to solid
            idx = random.choice(range(size))
            i, j = (line_indices, idx) if is_row else (idx, line_indices)
            if (i, j) not in path:
                grid[i, j] = 1

    for i in range(size):
        fix_line(i, is_row=True)
    for j in range(size):
        fix_line(j, is_row=False)
    # Final check
    final_zeros = np.sum(grid == 0)
    assert porosity_min * total_cells <= final_zeros <= porosity_max * total_cells, \
        f"Porosity out of bounds: {final_zeros/total_cells:.2f}"
    for (i, j) in path:
        assert grid[i, j] == 0, "Path violated"
    return grid

# Example usage
if __name__ == "__main__":

    for index in range(5):
        structure = generate_structure()
        print("New Structure "+str(index+1)+":")
        print("Structure:")
        for row in structure:
            print(''.join(str(int(x)) for x in row))

        print('\n')
