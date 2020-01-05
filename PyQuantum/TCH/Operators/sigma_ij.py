import numpy as np


def sigma_ij(i, j, n_levels=2):
    i_ = np.zeros([n_levels, 1])
    i_[i][0] = 1

    j_ = np.zeros([1, n_levels])
    j_[0][j] = 1

    # print(i_)
    # print(j_)
    return i_.dot(j_)
