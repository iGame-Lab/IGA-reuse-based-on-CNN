import torch
import numpy as np
import math

from loss.utils import findSpan, dersBasisFunc

def computeInnerBasis(index, u, v, knot_vector, ctrlp_num, order_vector):
    n = ctrlp_num[index][1]
    p = order_vector[index][0]
    U = knot_vector[index][0]
    m = ctrlp_num[index][0]
    q = order_vector[index][1]
    V = knot_vector[index][1]
    uspan = findSpan(n - 1, p, u, U)
    ubasis = dersBasisFunc(uspan, u, p, 2, U)
    vspan = findSpan(m - 1, q, v, V)
    vbasis = dersBasisFunc(vspan, v, q, 2, V)
    ders = np.zeros((5, p + 1, q + 1))
    ders[0] = np.outer(vbasis[0], ubasis[1])
    ders[1] = np.outer(vbasis[1], ubasis[0])
    ders[2] = np.outer(vbasis[0], ubasis[2])
    ders[3] = np.outer(vbasis[2], ubasis[0])
    ders[4] = np.outer(vbasis[1], ubasis[1])
    allDers = np.zeros((5, m, n))
    allDers[:, vspan - q:vspan + 1, uspan - p:uspan + 1] = ders 
    uvbasis = np.zeros((m, n))
    uvbasis[vspan - q:vspan + 1, uspan - p:uspan + 1] = np.outer(vbasis[0], ubasis[0])
    return allDers.tolist(), uvbasis.tolist()

def computeBdBasis(index, u, v, knot_vector, ctrlp_num, order_vector):
    U = knot_vector[index][0]
    V = knot_vector[index][1]
    p = order_vector[index][0]
    q = order_vector[index][1]
    m = ctrlp_num[index][0]
    n = ctrlp_num[index][1]
    uspan = findSpan(n - 1, p, u, U)
    ubasis = dersBasisFunc(uspan, u, p, 0, U)
    vspan = findSpan(m - 1, q, v, V)
    vbasis = dersBasisFunc(vspan, v, q, 0, V)
    uvbasis = np.zeros((m, n))
    uvbasis[vspan - q:vspan + 1, uspan - p:uspan + 1] = np.outer(vbasis[0], ubasis[0])
    return uvbasis.tolist()

def computeAllSpanBasis(sample_num, patch_num, knot_vector, ctrlp_num, order_vector, boundary, device):
    u_samples = np.linspace(0, 1, sample_num)
    bd_patch = []
    bd_basis = []
    inner_patch = []
    inner_basis_ders = []
    inner_basis = []
    for k in range(patch_num):
        for i in range(sample_num):
            for j in range(sample_num):
                u = u_samples[j]
                v = u_samples[i]
                if u == 0 and boundary[k][0] or u == 1 and boundary[k][1] \
                    or v == 0 and boundary[k][2] or v == 1 and boundary[k][3]:
                    basis = computeBdBasis(k, u, v, knot_vector, ctrlp_num, order_vector)
                    bd_patch.append(k)
                    bd_basis.append(basis)
                elif u != 0 and v != 0 and u != 1 and v != 1:
                    basis_ders, basis = computeInnerBasis(k, u, v, knot_vector, ctrlp_num, order_vector)
                    inner_patch.append(k)
                    inner_basis_ders.append(basis_ders)
                    inner_basis.append(basis)
    return torch.tensor(bd_patch, device=device), \
           torch.tensor(bd_basis, dtype=torch.float32, device=device), \
           torch.tensor(inner_patch, device=device), \
           torch.tensor(inner_basis_ders, dtype=torch.float32, device=device), \
           torch.tensor(inner_basis, dtype=torch.float32, device=device)

def matrixMap(input, mapRecords, output, patch_num, ctrlp_num, device, dim=2):
    batch_size = input.size(0)
    ctrlp = torch.zeros((batch_size, 2, mapRecords[0].size(0)), device=device)
    coefs = torch.zeros((batch_size, mapRecords[0].size(0)), device=device)
    for i in range(batch_size):
        ctrlp[i, 0] = input[i, 1, mapRecords[i, :, 0], mapRecords[i, :, 1]]
        ctrlp[i, 1] = input[i, 2, mapRecords[i, :, 0], mapRecords[i, :, 1]]
        coefs[i] = output[i, 0, mapRecords[i, :, 0], mapRecords[i, :, 1]]
    return torch.reshape(ctrlp, (batch_size, dim, patch_num, ctrlp_num[0][0], ctrlp_num[0][1])), \
            torch.reshape(coefs, (batch_size, patch_num, ctrlp_num[0][0], ctrlp_num[0][1]))

def innerError(ctrlp, coefs, inner_patch, inner_basis_ders, inner_basis, pdeType, ctrlp_num, device):
    ctrlp_expand = ctrlp[:, :, inner_patch]
    coefs_expand = coefs[:, inner_patch]
    inner_basis_ders_expand = inner_basis_ders.repeat((ctrlp.size(0), 1, 1, 1, 1))
    # xu, yu, xv, yv, xuu, yuu, xvv, yvv, xuv, yuv
    surface = torch.zeros((ctrlp.size(0), inner_patch.size(0), 10), device=device)
    for i in range(5):
        for j in range(2):
            surface[:, :, i * 2 + j] = torch.sum(ctrlp_expand[:, j] * inner_basis_ders_expand[:, :, i], dim=(2, 3))
    surface = torch.repeat_interleave(surface[:, :, None, :], repeats=int(ctrlp_num[0][0] * ctrlp_num[0][1]), dim=3). \
        reshape(ctrlp.size(0), inner_patch.size(0), 10, int(ctrlp_num[0][0]), int(ctrlp_num[0][1]))
    J2_deter = surface[:, :, 0] * surface[:, :, 3] - surface[:, :, 1] * surface[:, :, 2]
    J3 = torch.zeros((ctrlp.size(0), inner_patch.size(0), 3, 3, int(ctrlp_num[0][0]), int(ctrlp_num[0][1])), device=device)
    J3[:, :, 0, 0] = surface[:, :, 0] ** 2
    J3[:, :, 0, 1] = 2 * surface[:, :, 0] * surface[:, :, 1]
    J3[:, :, 0, 2] = surface[:, :, 1] ** 2
    J3[:, :, 1, 0] = surface[:, :, 0] * surface[:, :, 2]
    J3[:, :, 1, 1] = surface[:, :, 0] * surface[:, :, 3] + surface[:, :, 2] * surface[:, :, 1]
    J3[:, :, 1, 2] = surface[:, :, 1] * surface[:, :, 3]
    J3[:, :, 2, 0] = surface[:, :, 2] ** 2
    J3[:, :, 2, 1] = 2 * surface[:, :, 2] * surface[:, :, 3]
    J3[:, :, 2, 2] = surface[:, :, 3] ** 2
    J3_deter = J3[:, :, 0, 0] * J3[:, :, 1, 1] * J3[:, :, 2, 2] + \
               J3[:, :, 0, 1] * J3[:, :, 1, 2] * J3[:, :, 2, 0] + \
               J3[:, :, 0, 2] * J3[:, :, 1, 0] * J3[:, :, 2, 1] - \
               J3[:, :, 0, 2] * J3[:, :, 1, 1] * J3[:, :, 2, 0] - \
               J3[:, :, 0, 0] * J3[:, :, 1, 2] * J3[:, :, 2, 1] - \
               J3[:, :, 0, 1] * J3[:, :, 1, 0] * J3[:, :, 2, 2]
    Bx = (inner_basis_ders_expand[:, :, 0] * surface[:, :, 3] + \
          inner_basis_ders_expand[:, :, 1] * (-surface[:, :, 1]))/J2_deter
    By = (inner_basis_ders_expand[:, :, 0] * (-surface[:, :, 2]) + \
          inner_basis_ders_expand[:, :, 1] * surface[:, :, 0])/J2_deter
    left1 = inner_basis_ders_expand[:, :, 2] - Bx * surface[:, :, 4] - By * surface[:, :, 5]
    left2 = inner_basis_ders_expand[:, :, 4] - Bx * surface[:, :, 8] - By * surface[:, :, 9]
    left3 = inner_basis_ders_expand[:, :, 3] - Bx * surface[:, :, 6] - By * surface[:, :, 7]
    Bxx = (left1 * (J3[:, :, 1, 1] * J3[:, :, 2, 2] - J3[:, :, 2, 1] * J3[:, :, 1, 2]) + \
          left2 * (J3[:, :, 2, 1] * J3[:, :, 0, 2] - J3[:, :, 0, 1] * J3[:, :, 2, 2]) + \
          left3 * (J3[:, :, 0, 1] * J3[:, :, 1, 2] - J3[:, :, 0, 2] * J3[:, :, 1, 1])) / J3_deter
    Byy = (left1 * (J3[:, :, 1, 0] * J3[:, :, 2, 1] - J3[:, :, 2, 0] * J3[:, :, 1, 1]) + \
          left2 * (J3[:, :, 2, 0] * J3[:, :, 0, 1] - J3[:, :, 0, 0] * J3[:, :, 2, 1]) + \
          left3 * (J3[:, :, 0, 0] * J3[:, :, 1, 1] - J3[:, :, 0, 1] * J3[:, :, 1, 0])) / J3_deter

    if pdeType == 1:
        # f1
        error = torch.mean(abs(torch.sum(coefs_expand * (Bxx + Byy), dim=(2, 3)) + 100))
    elif pdeType == 2:
        # f2
        inner_basis_expand = inner_basis.repeat((ctrlp.size(0), 1, 1, 1))
        x = torch.sum(ctrlp_expand[:, 0] * inner_basis_expand, dim=(2, 3))
        error = torch.mean(abs(torch.sum(coefs_expand * (Bxx + Byy), dim=(2, 3)) - 100 * math.pi ** 2 * torch.sin(2 * math.pi * x)))
    else:
        # f3
        inner_basis_expand = inner_basis.repeat((ctrlp.size(0), 1, 1, 1))
        x = torch.sum(ctrlp_expand[:, 0] * inner_basis_expand, dim=(2, 3))
        y = torch.sum(ctrlp_expand[:, 1] * inner_basis_expand, dim=(2, 3))
        error = torch.mean(abs(torch.sum(coefs_expand * (Bxx + Byy), dim=(2, 3)) -
                100 * math.pi ** 2 * torch.sin(2 * math.pi * x) * torch.sin(2 * math.pi * y)))
    return error

def bdError(coefs, bd_patch, bd_basis):
    coefs_expand = coefs[:, bd_patch]
    bd_basis_expand = bd_basis.repeat((coefs.size(0), 1, 1, 1))
    bd_error = torch.mean(abs(torch.sum(coefs_expand * bd_basis_expand, dim=(2, 3))))
    return bd_error

def pde_loss(input, mapRecord, output, bd_patch, bd_basis, inner_patch, inner_basis_ders, inner_basis, pdeType, ctrlp_num, patch_num, device):
    ctrlp, coefs = matrixMap(input, mapRecord, output, patch_num, ctrlp_num, device)
    inner_error = innerError(ctrlp, coefs, inner_patch, inner_basis_ders, inner_basis, pdeType, ctrlp_num, device)
    bd_error = bdError(coefs, bd_patch, bd_basis)
    return inner_error, bd_error

class PDETestError(torch.nn.Module):
    def __init__(self, device, patch_num, order_vector, knot_vector, ctrlp_num, boundary, sample_num=10, pdeType=1):
        super(PDETestError, self).__init__()
        self.sample_num = sample_num
        self.patch_num = patch_num
        self.order_vector = order_vector
        self.ctrlp_num = ctrlp_num
        self.knot_vector = knot_vector
        self.boundary = boundary
        self.pdeType = pdeType
        self.device = device
        self.bd_patch, self.bd_basis, self.inner_patch, self.inner_basis_ders, self.inner_basis = computeAllSpanBasis(self.sample_num, self.patch_num, self.knot_vector, self.ctrlp_num, self.order_vector, self.boundary, self.device)
        
    def forward(self, input, mapRecord, output):
        geoInputs = torch.zeros_like(output)
        geoInputs[:, 0] = input[:, 0]
        zero_template = torch.zeros_like(output)
        outputs_coefs = torch.where(geoInputs > 0, output, zero_template)
        inner_error, _ = pde_loss(input, mapRecord, outputs_coefs, self.bd_patch, self.bd_basis, self.inner_patch, self.inner_basis_ders, self.inner_basis, self.pdeType, self.ctrlp_num, self.patch_num, self.device)
        return inner_error
