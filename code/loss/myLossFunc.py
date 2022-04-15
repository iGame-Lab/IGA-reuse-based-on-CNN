from unicodedata import decimal
import torch
import numpy as np
from loss.utils import findSpan, dersBasisFunc

def computeBasis(index, u, v, knot_vector, order_vector, ctrlp_num):
    U = knot_vector[index][0]
    V = knot_vector[index][1]
    p = order_vector[index][0]
    q = order_vector[index][1]
    n = ctrlp_num[index][0]
    m = ctrlp_num[index][1]
    uspan = findSpan(n - 1, p, u, U)
    ubasis = dersBasisFunc(uspan, u, p, 0, U)
    vspan = findSpan(m - 1, q, v, V)
    vbasis = dersBasisFunc(vspan, v, q, 0, V)
    uvbasis = np.zeros((m, n))
    uvbasis[vspan - q:vspan + 1, uspan - p:uspan + 1] = np.outer(vbasis[0][:, None], ubasis[0][None, :])
    return uvbasis.tolist()

def computeAllSpanBasis(sample_num, patch_num, knot_vector, order_vector, ctrlp_num, device):
    u_samples = np.linspace(0, 1, sample_num) 
    patchIndex = []
    basis = []
    for k in range(patch_num):
        for i in range(sample_num):
            for j in range(sample_num):
                u = u_samples[j]
                v = u_samples[i]
                patchIndex.append(k)
                basis.append(computeBasis(k, u, v, knot_vector, order_vector, ctrlp_num))
    return torch.tensor(patchIndex, device=device), torch.tensor(basis, dtype=torch.float32, device=device)


def matrixMap(mapRecord, target, output, patch_num, ctrlp_num, device):
    batch_size = target.size(0)
    channel_num = target.size(1)
    targets_coefs = torch.zeros((batch_size, channel_num, mapRecord.size(1)), device=device)
    outputs_coefs = torch.zeros_like(targets_coefs, device=device)
    for i in range(batch_size):
        for j in range(channel_num):
            targets_coefs[i][j] = target[i, j, mapRecord[i, :, 0], mapRecord[i, :, 1]]
            outputs_coefs[i][j] = output[i, j, mapRecord[i, :, 0], mapRecord[i, :, 1]]
    return torch.reshape(targets_coefs, (batch_size, channel_num, patch_num, ctrlp_num[0][0], ctrlp_num[0][1])),\
           torch.reshape(outputs_coefs, (batch_size, channel_num, patch_num, ctrlp_num[0][0], ctrlp_num[0][1])),  

def solutionLoss(mapRecord, target, output, patchIndex, basis, patch_num, ctrlp_num, device):
    targets_coefs_temp, outputs_coefs_temp = matrixMap(mapRecord, target, output, patch_num, ctrlp_num, device)
    targets_coefs = targets_coefs_temp[:, :, patchIndex] 
    outputs_coefs = outputs_coefs_temp[:, :, patchIndex]
    basis_expand = basis.repeat(output.size(0), output.size(1), 1, 1, 1)
    true_sol = torch.sum(targets_coefs * basis_expand, dim=(3, 4))
    pred_sol = torch.sum(outputs_coefs * basis_expand, dim=(3, 4))
    error = torch.mean(abs(true_sol - pred_sol))
    return error

def coefsLoss(input, target, output, device):
    geoInput = torch.zeros_like(output, device=device)
    for i in range(output.size(1)):
        geoInput[:, i] = input[:, 0]
    zero_template = torch.zeros_like(output)
    outputs_coefs = torch.where(geoInput != 0, output, zero_template)
    count = torch.sum(geoInput)
    loss = torch.sum(abs(outputs_coefs - target)) / count
    return loss

class TotalLoss(torch.nn.Module):
    def __init__(self, device, patch_num, order_vector, knot_vector, ctrlp_num, w1=1, w2=0, sample_num=10):
        super(TotalLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.patch_num = patch_num
        self.order_vector = order_vector
        self.ctrlp_num = ctrlp_num
        self.knot_vector = knot_vector
        self.sample_num = sample_num
        self.device = device
        self.patchIndex, self.basis = computeAllSpanBasis(self.sample_num, self.patch_num, self.knot_vector, self.order_vector, self.ctrlp_num, self.device)

    def forward(self, input, mapRecord, target, output):
        loss1 = coefsLoss(input, target, output, self.device)
        loss2 = solutionLoss(mapRecord, target, output, self.patchIndex, self.basis, self.patch_num ,self.ctrlp_num, self.device)
        loss = self.w1 * loss1 + self.w2 * loss2
        return loss, loss1, loss2

########################################################################

def computeAllTestSpanBasis(sample_num, patch_num, knot_vector, order_vector, ctrlp_num, boundary, device):
    u_samples = np.linspace(0, 1, sample_num) 
    patchIndex = []
    basis = []
    for k in range(patch_num):
        for i in range(sample_num):
            for j in range(sample_num):
                u = u_samples[j]
                v = u_samples[i]
                if u == 0 and boundary[k][0] or u == 1 and boundary[k][1] or v == 0 and boundary[k][2] or v == 1 and boundary[k][3]:
                    continue
                patchIndex.append(k)
                basis.append(computeBasis(k, u, v, knot_vector, order_vector, ctrlp_num))
    return torch.tensor(patchIndex, device=device), torch.tensor(basis, dtype=torch.float32, device=device)

def solutionTestError(mapRecord, target, output, patchIndex, basis, patch_num, ctrlp_num, device):
    targets_coefs_temp, outputs_coefs_temp = matrixMap(mapRecord, target, output, patch_num, ctrlp_num, device)
    targets_coefs = targets_coefs_temp[:, :, patchIndex]
    outputs_coefs = outputs_coefs_temp[:, :, patchIndex]
    basis_expand = basis.repeat(output.size(0), output.size(1), 1, 1, 1)
    true_sol = torch.sum(targets_coefs * basis_expand, dim=(3, 4))
    pred_sol = torch.sum(outputs_coefs * basis_expand, dim=(3, 4))
    sol_error = true_sol - pred_sol
    e = 0.000001
    zero_template = torch.zeros_like(sol_error)
    relate_error = torch.where(abs(true_sol) > e, sol_error / true_sol, zero_template) 
    return torch.mean(abs(sol_error)), torch.mean(abs(relate_error))

def coefsTestError(target, output, device):
    zero_template = torch.zeros_like(target, device=device)
    outputs_coefs = torch.where(target != 0, output, zero_template)
    count = torch.sum(target != 0)
    error = torch.sum(abs(outputs_coefs - target)) / count
    return error, outputs_coefs

class TestLoss(torch.nn.Module):
    def __init__(self, data_select, device, patch_num, order_vector, knot_vector, ctrlp_num, boundary, sample_num=10):
        super(TestLoss, self).__init__()
        self.data_select = data_select
        self.patch_num = patch_num
        self.order_vector = order_vector
        self.ctrlp_num = ctrlp_num
        self.knot_vector = knot_vector
        self.boundary = boundary
        self.sample_num = sample_num
        self.device = device
        if self.data_select == "human":
            self.patchIndex, self.basis = computeAllTestSpanBasis(self.sample_num, self.patch_num, self.knot_vector, self.order_vector, self.ctrlp_num, self.boundary, self.device)
        else:
            self.patchIndex, self.basis = computeAllSpanBasis(self.sample_num, self.patch_num, self.knot_vector, self.order_vector, self.ctrlp_num, self.device)

    def forward(self, mapRecord, targets, outputs):
        coefs_error, outputs_coefs = coefsTestError(targets, outputs, self.device)
        sol_error, relate_sol_error = solutionTestError(mapRecord, targets, outputs_coefs, self.patchIndex, self.basis, self.patch_num, self.ctrlp_num, self.device)
        return coefs_error, sol_error, relate_sol_error

################################################################################

def plotSolutionLoss(mapRecord, target, output, patchIndex, basis, patch_num, ctrlp_num, device):
    targets_coefs_temp, outputs_coefs_temp = matrixMap(mapRecord, target, output, patch_num, ctrlp_num, device)
    targets_coefs = targets_coefs_temp[:, :, patchIndex]
    outputs_coefs = outputs_coefs_temp[:, :, patchIndex]
    basis_expand = basis.repeat(output.size(0), output.size(1), 1, 1, 1)
    true_sol = torch.sum(targets_coefs * basis_expand, dim=(3, 4))
    pred_sol = torch.sum(outputs_coefs * basis_expand, dim=(3, 4))
    sol_error = true_sol - pred_sol
    e = 0.000001
    zero_template = torch.zeros_like(sol_error)
    relate_error = torch.where(abs(true_sol) > e, sol_error / true_sol, zero_template)
    return true_sol, pred_sol, abs(sol_error), abs(relate_error)

class TestPlot(torch.nn.Module):
    def __init__(self, device, patch_num, order_vector, knot_vector, ctrlp_num, sample_num=10):
        super(TestPlot, self).__init__()
        self.patch_num = patch_num
        self.order_vector = order_vector
        self.ctrlp_num = ctrlp_num
        self.knot_vector = knot_vector
        self.sample_num = sample_num
        self.device = device
        self.patchIndex, self.basis = computeAllSpanBasis(self.sample_num, self.patch_num, self.knot_vector, self.order_vector, self.ctrlp_num, self.device)

    def forward(self, mapRecord, target, output):
        _, outputs_coefs = coefsTestError(target, output, self.device)
        true_sol, pred_sol, sol_error, relate_error = plotSolutionLoss(mapRecord, target, outputs_coefs, self.patchIndex, self.basis, self.patch_num, self.ctrlp_num, self.device)
        return true_sol, pred_sol, sol_error, relate_error
