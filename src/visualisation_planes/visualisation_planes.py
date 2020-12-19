import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import torch
import torch.optim as optim
# from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib


def SingleChangeBasisMatrix(single_plane_parameter):
    a, b, c, _ = single_plane_parameter
    a, b, c = float(a), float(b), float(c)

    normal = np.array([a, b, c])
    normal = normal / np.sqrt(np.sum(normal ** 2))

    if sum(normal == np.array([0, 0, 1])) != 3:
        basis_x = np.array([1, 0, 0])
        basis_y = np.array([0, 1, 0])
        basis_z = np.array([0, 0, 1])

        # Construct rotation matrix to align z-axis basis to plane normal
        # Need to add exception, if normal = [0, 0, 1]. don't do basis rotation
        v = np.cross(basis_z, normal)
        ssc = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.identity(3) + ssc + np.matmul(ssc, ssc) * \
            (1 - np.dot(normal, basis_z)) / (np.linalg.norm(v) ** 2)

        # Change basis to plane normal basis
        # plane equation in new basis: z = 0
        # plane normal basis in standard coordinate
        new_basis_x = np.array([np.matmul(R, basis_x)])
        new_basis_y = np.array([np.matmul(R, basis_y)])
        new_basis_z = np.array([np.matmul(R, basis_z)])
        new_basis_matrix = np.concatenate(
            (new_basis_x.T, new_basis_y.T, new_basis_z.T), axis=1)

        C_inv = np.linalg.inv(new_basis_matrix)

    else:
        C_inv = np.identity(3)

    return C_inv, new_basis_matrix


def ChangeBasisMatrix(plane_parameters):
    # Input: Plane parameters (Lx4)
    # Output: Change of basis matrices (L x 3 x 3)
    L = len(plane_parameters)
    mat = SingleChangeBasisMatrix(plane_parameters[0])

    for i in range(1, L):
        mat = np.vstack((mat, SingleChangeBasisMatrix(plane_parameters[i])))

    mat = mat.reshape((L, 3, 3)).T
    mat = torch.tensor(mat, device='cuda')
    return mat


# Points in original coordinates

plane_param_np = np.array([[0.5, 0.5, 0.5, 0.2], [1, 0, 0, 2], [1, 4, 2, 5]])
plane_parameters = torch.tensor(plane_param_np)
print(plane_parameters)
p = torch.randn([3, 100, 3])
p = p[:, 0:100, :]  # Take 100 points
p = p.to('cpu')
p = p.numpy()
p = p[0]  # 100 points in first batch

# Change of basis matrix
C_inv, C = SingleChangeBasisMatrix(plane_parameters[0])

# p_project is in plane basis
p_project = np.matmul(p, C_inv)  # First plane
p_project[:, 2] = 0  # Project z coordinates to "ground"

# Convert back to earth basis
p_project_earthbasis = np.matmul(C, p_project.T).T

# Visualize

fig = plt.figure()
ax = Axes3D(fig)
# ax = plt.axes(projection='3d')
# ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points
ax.scatter3D(p[:, 0], p[:, 1], p[:, 2], cmap='Greens')
ax.scatter3D(p_project_earthbasis[:, 0], p_project_earthbasis[:,
                                                              1], p_project_earthbasis[:, 2], color='red')

a, b, c, d = plane_parameters[0].numpy()


#define the square in xy which contains all points of reconstructed object for the plane drawing in this range
min_val, _ = torch.min(p, dim=0, keepdim=True)
max_val, _ = torch.max(p, dim=0, keepdim=True)
max_abs_width = max(abs(min_val[0][0].item()), max_val[0][0].item()) + 1
max_abs_height = max(abs(min_val[0][1].item()), max_val[0][1].item()) + 1
print(max_abs_width)

x = np.linspace(-max_abs_width, max_abs_width, 10)
y = np.linspace(-max_abs_height, max_abs_height, 10)

X, Y = np.meshgrid(x, y)
Z = (d - a*X - b*Y) / c


# fig = plt.figure()
# ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z, alpha=0.2)

plt.savefig('illust.jpg')
