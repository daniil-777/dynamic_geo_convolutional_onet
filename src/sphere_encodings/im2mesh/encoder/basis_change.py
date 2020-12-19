def SingleChangeBasisMatrix(single_plane_parameter):
    # single_plane_parameter - torcch.tensor[1*4] dtype = torch.float32
    # a, b, c, _ = single_plane_parameter[0],
    # a, b, c = float(a), float(b), float(c)

    normal = single_plane_parameter[0:3]
    normal = normal / torch.sqrt(torch.sum(normal ** 2))

    if sum(normal == torch.tensor([0, 0, 1], dtype=torch.float32)) != torch.tensor(
        3, dtype=torch.uint8
    ):
        basis_x = torch.tensor([1, 0, 0], dtype=torch.float32)
        basis_y = torch.tensor([0, 1, 0], dtype=torch.float32)
        basis_z = torch.tensor([0, 0, 1], dtype=torch.float32)

        # Construct rotation matrix to align z-axis basis to plane normal
        # Need to add exception, if normal = [0, 0, 1]. don't do basis rotation
        v = torch.cross(basis_z, normal)
        #         print(v[0].view(-1).shape)
        zero_tensor = torch.tensor(0, dtype=torch.float32)
        ssc = torch.tensor(
            [
                zero_tensor,
                -v[2],
                v[1],
                v[2],
                zero_tensor,
                -v[0],
                -v[1],
                v[0],
                zero_tensor,
            ]
        ).view(3, 3)
        R = (
            torch.eye(3)
            + ssc
            + torch.matmul(ssc, ssc)
            * (1 - torch.dot(normal, basis_z))
            / (torch.norm(v, p=2, dim=0) ** 2)
        )

        # Change basis to plane normal basis
        # plane equation in new basis: z = 0
        # plane normal basis in standard coordinate
        new_basis_x = torch.matmul(R, basis_x)
        new_basis_y = torch.matmul(R, basis_y)
        new_basis_z = torch.matmul(R, basis_z)
        b_x = torch.abs(new_basis_x).view(-1)
        b_y = torch.abs(new_basis_y).view(-1)
        p_dummy = torch.tensor([1, 1, 1], dtype=torch.float32)
        p_x = torch.dot(p_dummy, b_x) / torch.dot(b_x, b_x) * b_x
        p_y = torch.dot(p_dummy, b_y) / torch.dot(b_y, b_y) * b_y
        c_x = torch.norm(p_x, p=2, dim=0)
        c_y = torch.norm(p_y, p=2, dim=0)

        if c_x > c_y:
            norm_c = torch.tensor([c_x, c_x, c_x])
        else:
            norm_c = torch.tensor([c_y, c_y, c_y])
        # really cat wrt 1 dim?

        new_basis_matrix = torch.t(
            torch.cat(
                (
                    torch.transpose(new_basis_x, 0, -1),
                    torch.transpose(new_basis_y, 0, -1),
                    torch.transpose(new_basis_z, 0, -1),
                ),
                0,
            ).view(3, 3)
        )
        print("new_basis_shape", new_basis_matrix.shape)
        C_inv = torch.inverse(new_basis_matrix)
        C_inv = C_inv.contiguous().view(-1)

    else:
        C_inv = torch.eye(3).view(-1)
        norm_c = torch.ones((3,))

    C_inv_norm_c = torch.cat((C_inv, norm_c), dim=0).view(4, 3)

    return C_inv_norm_c


def ChangeBasisMatrix(plane_parameters):
    # Input: Plane parameters (Lx4) - torch.tensor dtype = torch.float32
    # Output: Change of basis matrices (L x 3 x 3)
    L = plane_parameters.shape[0]
    mat = SingleChangeBasisMatrix(plane_parameters[0])

    for i in range(1, L):
        # really cat wrt 0 dim?
        mat = torch.cat((mat, SingleChangeBasisMatrix(plane_parameters[i])), 0)
    mat = mat.view(L, 4, 3)
    #     mat = torch.transpose(mat.view((3, 3)),0,1)
    # mat = torch.tensor(mat, device='cuda').float()
    # mat = torch.tensor(mat).float()
    return mat

    if __name__ == "__main__":
        plane_parameters = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], dtype=torch.float32
        ).view(3, 4)
        result = ChangeBasisMatrix(plane_parameters)
