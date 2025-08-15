import os
import random
import numpy as np
from numpy import zeros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from mpl_toolkits.mplot3d import Axes3D

nx, ny, nz = 64, 64, 64
pixel_size = 0.003 # meter
n_data = 5000
min_perm = 10
max_perm = 200
min_bound = 2*4096 #points on the boundary
max_bound = 5*min_bound
permeability = []
cubes = []
clouds = []
porosities = []
corr_lens = []
spec_perims = []

############################
###### Functions ######
############################
def compute_correlation_length(a, pixel_size):
    a_float = a.astype(np.float64)
    phi = a_float.mean()
    a0 = a_float - phi
    fft_a = np.fft.fftn(a0)
    corr_raw = np.fft.ifftn(fft_a * np.conj(fft_a)).real
    corr_raw = np.fft.fftshift(corr_raw) / a.size
    center = tuple(s // 2 for s in corr_raw.shape)
    C0 = corr_raw[center]
    if C0 <= 0:
        return 0.0
    corr_norm = corr_raw / C0
    # radial averaging
    coords = [np.arange(-s // 2, s - s // 2) for s in corr_raw.shape]
    Z, Y, X = np.meshgrid(coords[2], coords[1], coords[0], indexing='xy')
    distances = np.sqrt(X**2 + Y**2 + Z**2)
    max_r = min(center)
    radii = np.arange(0, max_r + 1)
    corr_profile = np.zeros_like(radii, dtype=np.float64)
    for i, r in enumerate(radii):
        mask = (distances >= i) & (distances < i + 1)
        if np.any(mask):
            corr_profile[i] = corr_norm[mask].mean()
    # find first decay to 1/e
    thresh = 1 / np.e
    below = np.where(corr_profile <= thresh)[0]
    return (below[0] if below.size else radii[-1]) * pixel_size

##################################################################

for iii in range(n_data):
  file = open('/scratch/users/kashefi/DiffPore/Data/Set1/3Dporous'+str(iii+1)+'.dat.res','r')
  all_lines = file.readlines()
  line = all_lines[10]
  digit = len(line) - 19
  s = list(line)
  new = ""   
  for j in range(digit): 
    new += s[j+18+1]
  perm1 = float(new)       
  if perm1 == 0 or (perm1 > max_perm) or (perm1 < min_perm):
    file.close()
    continue
  file.close()

  file = open('/scratch/users/kashefi/DiffPore/Data/Set1/3Dporous'+str(iii+1)+'.dat.new','r')
  all_lines = file.readlines()
  count1 = 0
  all_node = zeros([nx,ny,nz],dtype='i')
  for i in range(nx):
    for j in range(ny):
      for b in range(nz):
        line = all_lines[count1]
        count1 += 1
        value = int(line.split()[0])
        if value == -1:
          all_node[i][j][b] = 0 #pore
          continue
        all_node[i][j][b] = 1 #grain

  file.close()
  tale = 0
  sub_cloud = []
  for i in range(nx):
    #if np.remainder(i,2) != 0:
    #  continue
    for j in range(ny):
      for b in range(nz): 
        value = all_node[i][j][b]
        if value == 0:
           if i==0 or j==0 or b==0 or i==nx-1 or j==ny-1 or b==nz-1:
             continue

           if all_node[i][j+1][b]==1 or all_node[i][j-1][b]==1 or all_node[i][j][b-1]==1 or all_node[i][j][b+1]==1:
              tale += 1
              sub_cloud.append([i,j,b])
              continue
        
  if tale < min_bound or tale > max_bound:
    continue
  
  indices = random.sample(range(tale), min_bound)
  clouds.append([sub_cloud[kk] for kk in indices])
  permeability.append(perm1)
  cubes.append(all_node)

  print("number of points on boundary (i slice): ",tale)
  print("permeability: ",perm1)

  # Compute porosity
  phi = np.mean(all_node == 0)
  # Compute correlation length
  corr_len = compute_correlation_length(all_node, pixel_size)
  # Compute specific perimeter (surface area per volume)
  dx = (all_node[1:, :, :] != all_node[:-1, :, :]).sum()
  dy = (all_node[:, 1:, :] != all_node[:, :-1, :]).sum()
  dz = (all_node[:, :, 1:] != all_node[:, :, :-1]).sum()
  surface_area = (dx + dy + dz) * (pixel_size ** 2)
  volume = all_node.size * (pixel_size ** 3)
  spec_perim = surface_area / volume
  
  porosities.append(phi)
  corr_lens.append(corr_len)
  spec_perims.append(spec_perim)

print('number of valid porous media is equal to ',len(permeability))

current_dir = os.getcwd()

for idx, cube in enumerate(cubes, start=1):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    filled = np.ones(cube.shape, dtype=bool)
    colors = np.empty(cube.shape + (4,), dtype=float)
    colors[cube == 0] = [1.0, 0.0, 0.0, 1.0]
    colors[cube == 1] = [0.0, 0.0, 1.0, 1.0]
    ax.voxels(filled, facecolors=colors, edgecolors='none', shade=False)
    ax.set_axis_off()
    ax.set_box_aspect(cube.shape)
    filename = os.path.join(current_dir, f"cube{idx}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

for idx, cloud in enumerate(clouds, start=1):
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(projection='3d')

    pts = np.array(cloud)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],s=2,c='b')
            
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_box_aspect((nx, ny, nz))

    filename = os.path.join(current_dir, f"pointcloud{idx}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

#### Save the results ####
np.save('cubes.npy', np.stack(cubes))
np.save('clouds.npy', np.stack(clouds))
np.save('permeability.npy', np.array(permeability))
np.save('porosity.npy', np.array(porosities))
np.save('corr_length.npy', np.array(corr_lens))
np.save('spec_perim.npy', np.array(spec_perims))
