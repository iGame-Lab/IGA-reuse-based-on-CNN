import numpy as np
import torch

def findSpan(n, p, u, U):
    if u == U[n+1]:
        return n
    low = p
    high = n + 1
    mid = int((low + high) / 2)
    while (u < U[mid] or u >= U[mid + 1]):
        if (u < U[mid]):
            high = mid
        else:
            low = mid
        mid = int((low + high) / 2)
    return mid

def dersBasisFunc(i, u, p, n, U):
    ders = np.zeros((n + 1, p + 1))
    ndu = np.zeros((p + 1, p + 1))
    a = np.zeros((2, p + 1))
    left = np.zeros((p + 1))
    right = np.zeros((p + 1))
    ndu[0][0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j][r] = right[r + 1] + left[j - r]
            if (ndu[j][r] == 0):
                print(i, j, r, u, U[i + 1 - j], U[i + j])
            temp = ndu[r][j - 1] / ndu[j][r]
            ndu[r][j] = saved + temp * right[r + 1]
            saved = temp * left[j - r]
        ndu[j][j] = saved

    for j in range(p + 1):
        ders[0][j] = ndu[j][p]

    # computes the derivatives
    for r in range(p + 1):
        s1 = 0
        s2 = 1
        a[0][0] = 1.0
        for k in range(1, n + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if (r >= k):
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                d = a[s2][0] * ndu[rk][pk]
            j1 = -rk
            if (rk >= -1):
                j1 = 1
            j2 = p - r
            if r - 1 <= pk:
                j2 = k - 1
            for j in range(j1, j2 + 1):
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                d += a[s2][j] * ndu[rk + j][pk]
            if r <= pk:
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                d += a[s2][k] * ndu[r][pk]
            ders[k][r] = d
            j = s1
            s1 = s2
            s2 = j

    r =  p
    for k in range(1, n + 1):
        for j in range(p + 1):
            ders[k][j] *= r
        r *= (p - k)

    return ders
