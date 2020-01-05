# =================================================== DESCRIPTION =====================================================
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# =================================================== DESCRIPTION =====================================================



# =================================================== EXAMPLES ========================================================
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# =================================================== EXAMPLES ========================================================



# =====================================================================================================================
# system
from math import sqrt
# ---------------------------------------------------------------------------------------------------------------------
# scientific
import numpy as np
from numpy import matrix, complex128
from scipy.sparse import lil_matrix, csc_matrix
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.Tools.Matrix import Matrix
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- SIGMA_X ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def sigma_x():
    return matrix([[0, 1], [1, 0]], dtype=complex128)
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- SIGMA_Y ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def sigma_y():
    return matrix([[0, complex(0, -1)], [complex(0, 1), 0]], dtype=complex128)
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- SIGMA_Z ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def sigma_z():
    return matrix([[1, 0], [0, -1]], dtype=complex128)
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- HADAMARD ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def Hadamard():
    return 1.0 / sqrt(2) * matrix([[1, 1], [1, -1]], dtype=complex128)
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- OPERATOR_A3 ------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def operator_a3(H, ph_type=1):
    H_size = H.size
    size = H_size

    data = np.array([np.zeros(size) for i in range(size)])

    if ph_type == 1:
        for k_from, v_from in enumerate(H.states):
            ph = v_from[0]

            if ph > 0:
                to_state = [ph - 1, v_from[1]] + [v_from[2]] + [[v_from[3][0]+1, v_from[3][1]]]
                print(v_from)
                # print(v_from[1])
                # print(v_from[2])
                # print(v_from[3])
                print(to_state)
                # exit(0)
                for k_to, v_to in enumerate(H.states):
                    if to_state == v_to:
                        data[k_to][k_from] = sqrt(ph)
                # data[k_to][k_from] = sqrt(ph)
    elif ph_type == 2:
        for k_from, v_from in enumerate(H.states):
            ph = v_from[1]

            if ph > 0:
                to_state = [v_from[0], ph - 1] + [v_from[2]] + [[v_from[3][0], v_from[3][1]+1]]

                # to_state = [v_from[0], ph - 1] + v_from[2:]

                for k_to, v_to in enumerate(H.states):
                    if to_state == v_to:
                        data[k_to][k_from] = sqrt(ph)
    a = Matrix(H.size, H.size, dtype=np.complex128)
    a.size = H.size
    a.data = lil_matrix(data, dtype=np.complex128)

    return a

def operator_a3all(H):
    H_size = H.size
    size = H_size

    data = np.array([np.zeros(size) for i in range(size)])

    # ph_type = 1
    for k_from, v_from in enumerate(H.states):
        ph = v_from[0]

        if ph > 0:
            to_state = [ph - 1, v_from[1]] + [v_from[2]] + [[v_from[3][0]+1, v_from[3][1]]]
                # print(v_from[1])
                # print(v_from[2])
                # print(v_from[3])
                # print(to_state)
                # exit(0)
            for k_to, v_to in enumerate(H.states):
                if to_state == v_to:
                    data[k_to][k_from] = sqrt(ph)
                    print(v_from, '->', to_state)
                # data[k_to][k_from] = sqrt(ph)
    
    # ph_type = 2
    for k_from, v_from in enumerate(H.states):
        ph = v_from[1]

        if ph > 0:
            to_state = [v_from[0], ph - 1] + [v_from[2]] + [[v_from[3][0], v_from[3][1]+1]]

                # to_state = [v_from[0], ph - 1] + v_from[2:]

            for k_to, v_to in enumerate(H.states):
                if to_state == v_to:
                    data[k_to][k_from] = sqrt(ph)
                    # print(v_from, '->', to_state)

    # exit(0)
    a = Matrix(H.size, H.size, dtype=np.complex128)
    a.size = H.size
    a.data = lil_matrix(data, dtype=np.complex128)

    return a

def operator_a3all_new(H, ph_type=1):
    H_size = H.size
    size = H_size

    data = np.array([np.zeros(size) for i in range(size)])

    # ph_type = 1
    if ph_type == 1:
        for k_from, v_from in enumerate(H.states):
            ph = v_from[0]

            if ph > 0:
                to_state = [ph - 1, v_from[1]] + [v_from[2]] + [[v_from[3][0]+1, v_from[3][1]]]
                    # print(v_from[1])
                    # print(v_from[2])
                    # print(v_from[3])
                    # print(to_state)
                    # exit(0)
                for k_to, v_to in enumerate(H.states):
                    if to_state == v_to:
                        data[k_to][k_from] = sqrt(ph)
                        print(v_from, '->', to_state)
                # data[k_to][k_from] = sqrt(ph)
    
    # ph_type = 2
    elif ph_type == 2:
        for k_from, v_from in enumerate(H.states):
            ph = v_from[1]

            if ph > 0:
                to_state = [v_from[0], ph - 1] + [v_from[2]] + [[v_from[3][0], v_from[3][1]+1]]

                    # to_state = [v_from[0], ph - 1] + v_from[2:]

                for k_to, v_to in enumerate(H.states):
                    if to_state == v_to:
                        data[k_to][k_from] = sqrt(ph)
                        # print(v_from, '->', to_state)

    # exit(0)
    a = Matrix(H.size, H.size, dtype=np.complex128)
    a.size = H.size
    a.data = lil_matrix(data, dtype=np.complex128)

    return a
# def operator_a3(H, ph_type=1):
#     H_size = H.size
#     size = H_size

#     data = np.array([np.zeros(size) for i in range(size)])

#     if ph_type == 1:
#         for k_from, v_from in enumerate(H.states):
#             ph = v_from[0]

#             if ph > 0:
#                 to_state = [ph - 1, v_from[1]] + v_from[2:]

#                 for k_to, v_to in enumerate(H.states):
#                     if to_state == v_to:
#                         data[k_to][k_from] = sqrt(ph)
#     elif ph_type == 2:
#         for k_from, v_from in enumerate(H.states):
#             ph = v_from[1]

#             if ph > 0:
#                 to_state = [v_from[0], ph - 1] + v_from[2:]

#                 for k_to, v_to in enumerate(H.states):
#                     if to_state == v_to:
#                         data[k_to][k_from] = sqrt(ph)

#     a = Matrix(H.size, H.size, dtype=np.complex128)
#     a.size = H.size
#     a.data = lil_matrix(data, dtype=np.complex128)

#     # a_dense = a.data.toarray()

#     # for i in range(H.size):
#     #     for j in range(H.size):
#     #         if abs(a.data[i, j]) != 0:
#     #             print(i, j, abs(a.data[i, j]))
#     #             print(H.states[i], H.states[j])
#     #             print()
#     # for i in len()
#     # print(a.data[0,0])
#     # exit(0)
#     return a
#     # return sp.csc_matrix(np.matrix(a))
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- OPERATOR_A -------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def operator_a(H):
    H_size = H.size
    size = H_size

    data = np.array([np.zeros(size) for i in range(size)])

    for k_from, v_from in enumerate(H.states):
        ph = v_from[0]

        if ph > 0:
            # print("ph = ", ph)
            # print("from_state = ", v_from)

            to_state = [ph - 1] + v_from[1:]
            # print("to_state0 = ", to_state)

            for k_to, v_to in enumerate(H.states):
                if to_state == v_to:
                    # print("to_state = ", to_state)
                    # a[k_from][k_to] = sqrt(ph)
                    data[k_to][k_from] = sqrt(ph)

    # for i in range(H_size):
    #     for j in range(H_size):
    #         if data[i][j] != 0:
    #             print(H.states[j], " -> ", H.states[i], ": ", np.round(data[i][j], 3), sep="")

    a = Matrix(H.size, H.size, dtype=np.complex128)
    a.size = H.size
    a.data = lil_matrix(data, dtype=np.complex128)

    # a_dense = a.data.toarray()

    # for i in range(H.size):
    #     for j in range(H.size):
    #         if abs(a.data[i, j]) != 0:
    #             print(i, j, abs(a.data[i, j]))
    #             print(H.states[i], H.states[j])
    #             print()
    # for i in len()
    # print(a.data[0,0])
    # exit(0)
    return a
    # return sp.csc_matrix(np.matrix(a))
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- OPERATOR_ACROSSA -------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def operator_acrossa(H, m, n):
    op_a = operator_a(H, m, n)
    op_across = op_a.conj()

    matrix = Matrix(H.size, H.size, dtype=np.complex128)
    matrix.data = (op_across.data).dot(op_a.data)

    return matrix
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- OPERATOR_ACROSS --------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def operator_across(H, m, n):
    op_a = operator_a(H, m, n)
    op_across = op_a.conj()

    matrix = Matrix(H.size, H.size, dtype=np.complex128)
    matrix.data = op_across.data
    # matrix.print()
    # exit(0)

    return matrix
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- OPERATOR_AACROSS -------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def operator_aacross(H, m, n):
    op_a = operator_a(H, m, n)
    op_across = op_a.conj()

    matrix = Matrix(H.size, H.size, dtype=np.complex128)
    matrix.data = (op_a.data).dot(op_across.data)

    return matrix
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- OPERATOR_L -------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def operator_L(ro, lindblad):
    l = lindblad['l']
    L = lindblad['L'].data

    Lcross = L.getH()
    LcrossL = Lcross.dot(L)

    # print(L)
    # l = []

    # L = []
    # L_cross = []

    # LcrossL = []

    # for i in lindblad:
    #     L.append(i['L'].data)
    #     L_cross.append(i['L'].data.getH())
    #     l.append(i['l'])

    # a = op_a.data
    # across = a.getH()

    # print("cross")
    # for i in L_cross:
    #     m = Matrix(m=ro.m, n=ro.n, dtype=np.complex128)
    #     m.data = i
    #     m.print()
    # exit(0)

    def b(ro):
        nonlocal L, Lcross, LcrossL, l
        # nonlocal L, L_cross, l

        L1 = 0
        L2 = 0

        L_ro = Matrix(m=ro.m, n=ro.n, dtype=np.complex128)

        # for i in range(len(l)):
        L1 += (L.dot(ro.data)).dot(Lcross)
        # L1 += (L[i].dot(ro.data)).dot(L_cross[i])

        L2 = np.dot(ro.data, LcrossL) + np.dot(LcrossL, ro.data)
        # L2 = np.dot(np.dot(ro.data, L_cross[i]), L[i]) + np.dot(np.dot(L_cross[i], L[i]), ro.data)

        # L_ro.data += csc_matrix(L1 - 0.5 * L2)
        L_ro.data += l * csc_matrix(L1 - 0.5 * L2, dtype=np.complex128)
        # L_ro.data += l[i] * lil_matrix(L1 - 0.5 * L2, dtype=np.complex128)
        # print()
        # ro.print()
        # print()
        # L_ro.data /= l[0]
        # L_ro.print()
        # exit(0)
        return L_ro
        # nonlocal a, across

        # L1 = (a.dot(ro.data)).dot(across)

        # L2 = np.dot(np.dot(ro.data, across), a) + np.dot(np.dot(across, a), ro.data)

        # L = Matrix(m=np.shape(ro.data)[0], n=np.shape(ro.data)[0], dtype=np.complex128)

        # L.data = lil_matrix(L1 - 0.5 * L2, dtype=np.complex128)

        # return L

    return b
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# =====================================================================================================================
# def operator_L(ro, a, a_cross, across_a):
#     L1 = (a.dot(ro)).dot(a_cross)

#     L2 = np.dot(np.dot(ro, a_cross), a) + np.dot(np.dot(a_cross, a), ro)

#     L = Matrix(m=np.shape(ro)[0], n=np.shape(ro)[0], dtype=np.complex128)

#     L.data = lil_matrix(L1 - 0.5 * L2, dtype=np.complex128)

#     return L
# =====================================================================================================================
