import numpy as np
import torch
import h5py

# Load and concatenate all sets
all_cubes = []
all_perms = []
for i in range(1, 5):
    folder = f'/scratch/users/kashefi/DiffPore/Data/Set{i}cubes'
    cubes_i = np.load(f'{folder}/cubes.npy')
    perms_i = np.load(f'{folder}/permeability.npy')
    all_cubes.append(cubes_i)
    all_perms.append(perms_i)

cubes = np.concatenate(all_cubes, axis=0)   # shape: (N_total, D, H, W)
perms = np.concatenate(all_perms, axis=0)   # shape: (N_total,)

all_cubes = []
all_perms = []

# Save to HDF5
h5_path = '/scratch/users/kashefi/DiffPore/Data/all_data.h5'
with h5py.File(h5_path, 'w') as hf:
    hf.create_dataset('cubes', data=cubes)
    hf.create_dataset('perms',  data=perms)

