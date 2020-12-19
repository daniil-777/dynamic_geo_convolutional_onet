import os
import numpy as np
import trimesh
import glob
import imageio
import sys
import torch
from tqdm import tqdm
from os.path import join

item_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/ScanNet_depth14/scenes'
item_path = os.path.join(item_path, os.listdir(item_path)[0])

depth_path = join(item_path, 'depth')
depth_files = glob.glob(join(depth_path, '*.png'))
depth_files.sort()

fmin = np.inf 
fmax = -np.inf

for f in tqdm(depth_files):
    img = imageio.imread(f).astype(np.float32) / 1000.
    mmin, mmax = img.min(), img.max()
    if mmin < fmin:
        fmin = mmin
    if mmax > fmax:
        fmax = mmax


print('Final numbers: %.6f, %.6f' % (fmin, fmax))