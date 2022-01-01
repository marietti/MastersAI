import numba as nb
import scipy.io as sio
from pylab import *
from scipy import linalg
import numpy.matlib
import cProfile



def compmedDist(X):
    size1 = X.shape[0]
    Xmed = X

    G = sum((Xmed * Xmed), 1)
    Q = tile(G[:, newaxis], (1, size1))
    R = tile(G, (size1, 1))

    dists = Q + R - 2 * dot(Xmed, Xmed.T)
    dists = dists - tril(dists)
    dists = dists.reshape(size1 ** 2, 1, order='F').copy()
    return sqrt(0.5 * median(dists[dists > 0]))

@nb.njit(fastmath=True)
def lstsq_test(A, b):
    AA = A.T @ A
    bA = b @ A
    D, U = np.linalg.eigh(AA)
    Ap = (U * np.sqrt(D)).T
    bp = bA @ U / np.sqrt(D)
    return np.linalg.lstsq(Ap, bp, rcond=None)


def kernel_Gaussian(x, c, sigma):
    (d, nx) = x.shape
    (d, nc) = c.shape
    x2 = sum(x ** 2, 0)
    c2 = sum(c ** 2, 0)

    distance2 = tile(c2, (nx, 1)) + \
                tile(x2[:, newaxis], (1, nc)) \
                - 2 * dot(x.T, c)

    return exp(-distance2 / (2 * (sigma ** 2)))


def kernel_Gaussian_PY(x, c, sigma):
    (d, nx) = x.shape
    (d, nc) = c.shape
    x2 = sum(x ** 2, 0)
    c2 = sum(c ** 2, 0)

    distance2 = tile(c2, (nx, 1)) + \
                tile(x2[:, newaxis], (1, nc)) \
                - 2 * dot(x.T, c)

    return exp(-distance2 / (2 * (sigma ** 2)))


def R_ULSIF_PY(x_nu, x_de, x_re, alpha, sigma_list, lambda_list, b, fold):
    # x_nu: samples from numerator
    # x_de: samples from denominator
    # x_re: reference sample
    # alpha: alpha defined in relative density ratio
    # sigma_list, lambda_list: parameters for model selection
    # b: number of kernel basis
    # fold: number of fold for cross validation

    (d, n_nu) = x_nu.shape
    (d, n_de) = x_de.shape
    rand_index = permutation(n_nu)
    b = min(b, n_nu)
    # x_ce = x_nu[:,rand_index[0:b]]
    x_ce = x_nu[:, r_[0:b]]

    score_cv = zeros((size(sigma_list),
                      size(lambda_list)))

    cv_index_nu = permutation(n_nu)
    # cv_index_nu = r_[0:n_nu]
    cv_split_nu = floor(r_[0:n_nu] * fold / n_nu)
    cv_index_de = permutation(n_de)
    # cv_index_de = r_[0:n_de]
    cv_split_de = floor(r_[0:n_de] * fold / n_de)

    for sigma_index in r_[0:size(sigma_list)]:
        sigma = sigma_list[sigma_index]
        K_de = kernel_Gaussian_PY(x_de, x_ce, sigma).T
        K_nu = kernel_Gaussian_PY(x_nu, x_ce, sigma).T

        score_tmp = zeros((fold, size(lambda_list)))

        for k in r_[0:fold]:
            Ktmp1 = K_de[:, cv_index_de[cv_split_de != k]]
            Ktmp2 = K_nu[:, cv_index_nu[cv_split_nu != k]]

            Ktmp = alpha / Ktmp2.shape[1] * dot(Ktmp2, Ktmp2.T) + \
                   (1 - alpha) / Ktmp1.shape[1] * dot(Ktmp1, Ktmp1.T)

            mKtmp = mean(K_nu[:, cv_index_nu[cv_split_nu != k]], 1)

            for lambda_index in r_[0:size(lambda_list)]:
                lbd = lambda_list[lambda_index]

                thetat_cv = linalg.solve(Ktmp + lbd * eye(b), mKtmp)
                thetah_cv = thetat_cv

                score_tmp[k, lambda_index] = alpha * mean(
                    dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv) ** 2) / 2. \
                                             + (1 - alpha) * mean(
                    dot(K_de[:, cv_index_de[cv_split_de == k]].T, thetah_cv) ** 2) / 2. \
                                             - mean(dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv))

            score_cv[sigma_index, :] = mean(score_tmp, 0)

    score_cv_tmp = score_cv.min(1)
    lambda_chosen_index = score_cv.argmin(1)

    score = score_cv_tmp.min()
    sigma_chosen_index = score_cv_tmp.argmin()

    lambda_chosen = lambda_list[lambda_chosen_index[sigma_chosen_index]]
    sigma_chosen = sigma_list[sigma_chosen_index]

    K_de = kernel_Gaussian_PY(x_de, x_ce, sigma_chosen).T
    K_nu = kernel_Gaussian_PY(x_nu, x_ce, sigma_chosen).T

    coe = alpha * dot(K_nu, K_nu.T) / n_nu + \
          (1 - alpha) * dot(K_de, K_de.T) / n_de + \
          lambda_chosen * eye(b)
    var = mean(K_nu, 1)

    thetat = linalg.solve(coe, var)
    #    thetat=linalg.lstsq(coe,var)[0]
    #    linalg.cho_factor(coe,overwrite_a=True)
    #    linalg.cho_solve((coe,False), var, overwrite_b=True)
    #    thetat = var

    # thetah=maximum(0,thetat)
    thetah = thetat
    wh_x_de = dot(K_de.T, thetah).T
    wh_x_nu = dot(K_nu.T, thetah).T

    K_di = kernel_Gaussian_PY(x_re, x_ce, sigma_chosen).T
    wh_x_re = dot(K_di.T, thetah).T

    wh_x_de[wh_x_de < 0] = 0
    wh_x_re[wh_x_re < 0] = 0

    PE = mean(wh_x_nu) - 1. / 2 * (alpha * mean(wh_x_nu ** 2) +
                                   (1 - alpha) * mean(wh_x_de ** 2)) - 1. / 2

    return PE, wh_x_re, score


def kernel_gau(dist2, sigma):
    k = exp(-dist2 / (2 * sigma ** 2))
    return k


def R_ULSIF(x_de, x_nu, x_re, x_ce, alpha_rulsif=0.5, fold=5):
    [_, n_nu] = shape(x_nu)
    [_, n_de] = shape(x_de)

    nu_perm = np.array([[48, 41, 40, 4, 45, 17, 18, 3, 26, 9, 23, 27, 21, 11, 19, 0, 35, 49, 7, 25, 30, 38, 47, 29, 5,
                         12, 42, 16, 33, 39, 46, 32, 8, 28, 2, 1, 13, 10, 37, 36, 43, 22, 31, 20, 14, 34, 24, 44, 15,
                         6],
                        [23, 24, 1, 19, 39, 14, 9, 15, 25, 35, 33, 20, 12, 31, 45, 6, 21, 36, 8, 11, 13, 0, 43, 44, 29,
                         42, 5, 4, 46, 47, 30, 18, 37, 28, 26, 34, 27, 32, 41, 38, 10, 49, 48, 22, 7, 2, 40, 16, 3, 17],
                        [43, 30, 10, 14, 20, 18, 45, 40, 29, 3, 41, 15, 27, 16, 8, 7, 49, 33, 4, 44, 22, 32, 48, 2, 1,
                         39, 35, 36, 5, 21, 6, 17, 28, 47, 24, 19, 31, 23, 13, 38, 37, 11, 42, 26, 25, 34, 12, 0, 46,
                         9],
                        [45, 10, 39, 25, 7, 18, 48, 9, 6, 49, 14, 22, 27, 30, 43, 33, 31, 0, 47, 37, 15, 21, 28, 46, 23,
                         16, 24, 3, 11, 29, 38, 26, 44, 32, 20, 2, 5, 13, 4, 1, 36, 42, 34, 19, 41, 12, 40, 17, 35, 8],
                        [1, 9, 24, 37, 13, 10, 22, 16, 34, 36, 18, 48, 20, 23, 6, 14, 43, 45, 3, 28, 39, 40, 4, 2, 8,
                         12, 27, 25, 38, 15, 49, 32, 29, 30, 41, 44, 31, 26, 35, 11, 17, 21, 5, 42, 7, 47, 46, 0, 19,
                         33],
                        [5, 49, 41, 42, 43, 48, 33, 27, 46, 9, 34, 19, 8, 39, 22, 6, 18, 13, 1, 15, 35, 7, 11, 24, 4,
                         47, 45, 30, 40, 32, 23, 44, 2, 3, 0, 36, 17, 26, 14, 38, 20, 16, 10, 37, 25, 28, 31, 12, 21,
                         29],
                        [48, 41, 2, 27, 31, 21, 22, 10, 40, 23, 0, 24, 38, 1, 19, 15, 14, 4, 44, 43, 45, 39, 18, 9, 46,
                         11, 7, 37, 8, 5, 16, 26, 25, 3, 17, 47, 6, 29, 30, 36, 49, 12, 35, 32, 13, 28, 33, 42, 20, 34],
                        [21, 7, 22, 45, 33, 5, 46, 12, 47, 11, 1, 38, 16, 39, 29, 19, 4, 6, 3, 35, 18, 32, 0, 41, 8, 25,
                         15, 40, 31, 24, 36, 10, 23, 30, 27, 26, 44, 37, 20, 9, 13, 34, 28, 48, 43, 17, 42, 14, 49, 2],
                        [21, 9, 14, 43, 20, 30, 26, 0, 18, 28, 19, 29, 31, 48, 32, 41, 2, 33, 42, 36, 37, 3, 8, 47, 15,
                         35, 34, 7, 22, 11, 12, 5, 25, 40, 46, 13, 49, 27, 23, 10, 17, 39, 45, 24, 6, 16, 4, 38, 44, 1],
                        [20, 9, 48, 32, 12, 5, 24, 0, 18, 31, 19, 29, 40, 14, 38, 16, 17, 37, 28, 47, 44, 3, 13, 7, 15,
                         27, 45, 36, 22, 1, 34, 8, 4, 35, 11, 30, 33, 42, 26, 10, 41, 39, 43, 49, 2, 25, 21, 23, 46, 6],
                        [3, 13, 1, 37, 14, 47, 10, 12, 46, 11, 26, 27, 48, 24, 42, 17, 40, 18, 30, 45, 44, 36, 29, 6,
                         34, 33, 35, 31, 15, 9, 25, 21, 28, 4, 38, 49, 32, 23, 5, 7, 43, 8, 2, 16, 20, 41, 19, 22, 0,
                         39],
                        [27, 36, 4, 44, 46, 31, 39, 30, 42, 11, 23, 15, 2, 37, 28, 17, 48, 8, 19, 38, 20, 14, 12, 47,
                         34, 1, 16, 33, 6, 35, 29, 21, 32, 0, 40, 5, 43, 25, 9, 18, 49, 7, 10, 45, 24, 41, 3, 26, 22,
                         13],
                        [20, 34, 49, 19, 13, 47, 8, 0, 25, 22, 12, 15, 38, 42, 5, 35, 6, 14, 10, 37, 1, 31, 21, 7, 33,
                         43, 28, 36, 11, 48, 46, 18, 3, 32, 9, 17, 16, 26, 40, 4, 41, 44, 29, 27, 45, 39, 23, 30, 24,
                         2],
                        [22, 17, 38, 36, 18, 24, 0, 16, 25, 19, 46, 3, 47, 21, 20, 2, 33, 4, 5, 41, 8, 9, 14, 34, 39,
                         35, 10, 32, 23, 6, 31, 7, 29, 49, 42, 26, 11, 13, 1, 43, 40, 15, 27, 44, 37, 48, 45, 12, 28,
                         30],
                        [12, 0, 23, 6, 19, 8, 42, 10, 26, 38, 18, 44, 49, 37, 48, 45, 47, 41, 1, 29, 43, 32, 17, 20, 25,
                         9, 4, 34, 35, 11, 39, 21, 28, 3, 31, 24, 36, 22, 13, 2, 46, 33, 14, 5, 40, 7, 15, 27, 30, 16],
                        [31, 9, 18, 47, 44, 48, 23, 28, 12, 14, 30, 49, 21, 22, 20, 37, 32, 0, 15, 5, 13, 46, 29, 16,
                         36, 7, 38, 45, 10, 35, 17, 19, 41, 4, 1, 34, 43, 33, 11, 27, 42, 39, 40, 26, 6, 8, 3, 2, 25,
                         24],
                        [27, 49, 24, 13, 38, 37, 12, 1, 19, 8, 14, 48, 39, 6, 44, 43, 34, 25, 22, 35, 5, 26, 40, 30, 29,
                         20, 11, 10, 47, 0, 17, 42, 18, 28, 7, 36, 33, 21, 31, 2, 4, 45, 15, 32, 16, 46, 9, 3, 41, 23],
                        [41, 8, 32, 2, 31, 43, 48, 12, 5, 22, 45, 49, 33, 1, 46, 23, 19, 38, 6, 44, 21, 13, 14, 35, 29,
                         37, 39, 25, 11, 34, 3, 47, 18, 24, 40, 7, 26, 10, 42, 0, 16, 36, 4, 20, 9, 17, 28, 30, 15, 27],
                        [28, 41, 30, 38, 26, 45, 13, 19, 35, 5, 6, 27, 0, 44, 22, 9, 34, 18, 31, 16, 36, 10, 39, 4, 7,
                         33, 15, 29, 48, 21, 20, 11, 46, 42, 14, 8, 23, 1, 40, 2, 32, 17, 49, 47, 12, 37, 3, 43, 25,
                         24],
                        [23, 45, 16, 28, 43, 25, 26, 7, 14, 5, 1, 27, 32, 39, 37, 49, 9, 22, 46, 8, 0, 15, 10, 24, 31,
                         38, 34, 4, 18, 40, 6, 47, 42, 29, 30, 12, 17, 48, 41, 3, 44, 2, 33, 19, 36, 21, 35, 13, 20,
                         11],
                        [21, 24, 3, 29, 40, 34, 41, 31, 17, 35, 37, 6, 43, 12, 0, 48, 13, 7, 27, 32, 36, 22, 23, 8, 15,
                         19, 1, 30, 18, 49, 46, 9, 42, 10, 11, 28, 38, 2, 4, 5, 33, 16, 47, 26, 25, 14, 20, 45, 44, 39],
                        [41, 33, 7, 17, 24, 45, 39, 10, 29, 43, 27, 18, 2, 42, 0, 23, 31, 9, 28, 6, 12, 22, 19, 11, 26,
                         8, 36, 13, 3, 46, 5, 4, 35, 1, 47, 37, 48, 15, 38, 25, 30, 21, 14, 34, 49, 16, 40, 44, 32, 20],
                        [37, 26, 28, 12, 35, 36, 44, 43, 2, 47, 40, 17, 19, 45, 0, 48, 49, 32, 10, 16, 13, 20, 30, 15,
                         6, 22, 27, 9, 33, 24, 1, 39, 23, 42, 7, 38, 25, 4, 46, 41, 3, 29, 31, 8, 21, 18, 5, 34, 14,
                         11],
                        [18, 40, 22, 44, 10, 14, 4, 24, 28, 17, 31, 3, 34, 16, 19, 41, 12, 5, 13, 8, 43, 46, 6, 11, 30,
                         39, 7, 37, 29, 1, 45, 35, 48, 20, 32, 15, 33, 9, 38, 47, 26, 42, 49, 25, 21, 23, 0, 27, 2, 36],
                        [29, 33, 42, 20, 23, 28, 35, 11, 43, 25, 16, 15, 38, 7, 4, 2, 14, 10, 1, 32, 48, 8, 18, 6, 3,
                         47, 17, 37, 27, 13, 34, 26, 49, 22, 45, 9, 36, 44, 19, 40, 31, 30, 46, 21, 12, 24, 41, 0, 39,
                         5]])
    nu_index = 0
    de_index = 0
    de_perm = np.array([[22, 45, 18, 11, 26, 20, 47, 35, 39, 34, 14, 0, 4, 32, 48, 12, 46, 49, 28, 6, 33, 2, 42, 27, 44,
                         30, 19, 10, 43, 3, 16, 38, 37, 5, 41, 7, 8, 9, 23, 1, 29, 24, 15, 40, 25, 36, 31, 13, 21, 17],
                        [43, 40, 27, 35, 17, 23, 15, 10, 34, 16, 18, 13, 48, 28, 31, 41, 36, 11, 20, 37, 4, 1, 29, 38,
                         22, 19, 26, 49, 12, 46, 39, 3, 7, 47, 9, 30, 32, 8, 0, 33, 45, 42, 2, 14, 6, 24, 21, 44, 5,
                         25],
                        [32, 2, 42, 0, 24, 3, 18, 5, 35, 19, 9, 34, 20, 37, 44, 30, 39, 36, 21, 25, 27, 48, 45, 31, 16,
                         15, 38, 11, 4, 26, 46, 29, 13, 28, 10, 1, 8, 43, 14, 23, 7, 6, 17, 22, 41, 47, 40, 12, 49, 33],
                        [47, 7, 39, 8, 22, 1, 18, 24, 41, 34, 40, 23, 25, 36, 9, 17, 44, 49, 12, 0, 21, 26, 28, 3, 30,
                         27, 10, 16, 2, 48, 19, 32, 6, 11, 42, 37, 46, 35, 38, 4, 5, 14, 20, 45, 13, 43, 29, 31, 33,
                         15],
                        [44, 19, 31, 17, 18, 5, 29, 41, 28, 16, 46, 43, 3, 4, 40, 49, 22, 36, 20, 47, 21, 8, 35, 39, 14,
                         32, 33, 1, 12, 10, 0, 42, 24, 9, 7, 11, 37, 48, 30, 13, 38, 45, 23, 25, 34, 2, 26, 6, 27, 15],
                        [8, 34, 42, 45, 24, 35, 48, 47, 49, 39, 15, 22, 44, 13, 37, 46, 16, 12, 40, 1, 18, 33, 31, 4,
                         21, 10, 43, 25, 41, 26, 23, 3, 27, 20, 30, 14, 5, 38, 19, 0, 36, 9, 6, 28, 17, 32, 2, 29, 7,
                         11],
                        [45, 43, 4, 22, 41, 48, 19, 31, 9, 38, 25, 5, 20, 35, 47, 26, 39, 42, 14, 37, 49, 7, 24, 2, 21,
                         15, 44, 3, 16, 29, 23, 11, 0, 34, 36, 6, 28, 12, 33, 32, 17, 18, 1, 40, 46, 8, 27, 30, 10, 13],
                        [15, 23, 37, 33, 39, 3, 40, 4, 1, 31, 25, 22, 28, 11, 35, 42, 34, 6, 16, 21, 17, 26, 38, 45, 48,
                         30, 36, 27, 9, 19, 43, 5, 29, 7, 49, 0, 18, 46, 2, 12, 10, 8, 47, 24, 32, 41, 20, 14, 44, 13],
                        [31, 4, 47, 43, 30, 10, 12, 1, 49, 46, 38, 2, 32, 17, 22, 33, 21, 15, 25, 11, 3, 8, 6, 24, 28,
                         29, 23, 37, 45, 18, 26, 20, 9, 34, 44, 5, 16, 41, 27, 42, 13, 40, 39, 48, 19, 14, 36, 35, 7,
                         0],
                        [31, 8, 46, 25, 48, 22, 7, 16, 24, 10, 20, 3, 43, 41, 18, 40, 27, 1, 12, 21, 36, 29, 6, 9, 42,
                         39, 47, 15, 38, 5, 37, 30, 0, 23, 4, 28, 26, 11, 14, 34, 33, 49, 44, 35, 13, 32, 19, 17, 45,
                         2],
                        [2, 6, 27, 39, 26, 22, 37, 20, 46, 16, 31, 21, 24, 33, 10, 15, 11, 30, 32, 7, 47, 38, 41, 13,
                         44, 23, 40, 34, 28, 14, 36, 12, 1, 8, 9, 18, 5, 29, 0, 43, 17, 3, 45, 25, 35, 49, 4, 48, 19,
                         42],
                        [15, 6, 41, 32, 39, 47, 16, 29, 27, 21, 25, 8, 42, 19, 23, 22, 40, 24, 33, 26, 3, 12, 31, 20, 4,
                         1, 0, 36, 46, 10, 13, 18, 9, 11, 37, 43, 44, 34, 35, 49, 14, 2, 45, 28, 38, 48, 30, 5, 7, 17],
                        [46, 28, 18, 21, 12, 30, 32, 25, 41, 10, 35, 0, 29, 6, 49, 48, 20, 11, 42, 4, 36, 16, 40, 17,
                         39, 27, 1, 26, 47, 37, 43, 31, 45, 5, 23, 13, 3, 34, 38, 9, 24, 14, 44, 33, 22, 19, 15, 2, 7,
                         8],
                        [37, 34, 47, 48, 39, 30, 44, 49, 14, 4, 42, 35, 11, 20, 22, 31, 17, 6, 25, 5, 21, 33, 41, 0, 12,
                         3, 13, 15, 43, 32, 46, 1, 8, 40, 45, 26, 16, 9, 29, 10, 19, 24, 23, 28, 2, 36, 18, 38, 27, 7],
                        [16, 14, 38, 33, 30, 12, 2, 7, 21, 0, 18, 32, 20, 17, 43, 15, 25, 45, 10, 4, 1, 29, 19, 5, 46,
                         27, 13, 49, 6, 3, 35, 31, 26, 8, 28, 42, 22, 41, 39, 40, 9, 37, 44, 11, 48, 47, 23, 24, 34,
                         36],
                        [4, 42, 10, 14, 24, 48, 40, 23, 38, 35, 29, 30, 45, 3, 34, 6, 26, 31, 21, 37, 33, 9, 36, 1, 39,
                         46, 15, 49, 5, 47, 43, 22, 16, 27, 41, 19, 11, 0, 17, 2, 18, 20, 44, 12, 8, 32, 7, 28, 13, 25],
                        [22, 23, 3, 35, 14, 18, 17, 42, 7, 44, 12, 39, 5, 38, 13, 32, 28, 26, 9, 37, 47, 8, 25, 19, 49,
                         29, 21, 16, 4, 1, 36, 33, 15, 11, 30, 48, 40, 43, 27, 46, 24, 10, 6, 0, 31, 20, 2, 34, 45, 41],
                        [35, 40, 32, 36, 46, 47, 28, 48, 30, 39, 6, 29, 1, 0, 37, 27, 2, 18, 42, 11, 22, 5, 8, 17, 45,
                         44, 38, 31, 14, 4, 26, 25, 49, 19, 10, 34, 21, 41, 23, 43, 13, 15, 9, 12, 3, 16, 20, 33, 24,
                         7],
                        [24, 27, 9, 40, 30, 29, 4, 15, 38, 39, 0, 43, 32, 48, 49, 41, 33, 2, 36, 34, 35, 26, 3, 16, 11,
                         21, 22, 31, 37, 17, 6, 45, 7, 44, 13, 28, 1, 12, 42, 8, 5, 46, 47, 14, 19, 23, 25, 20, 18, 10],
                        [26, 16, 8, 3, 10, 12, 45, 17, 36, 4, 20, 29, 2, 48, 6, 40, 37, 35, 19, 5, 15, 18, 41, 24, 49,
                         13, 33, 38, 9, 27, 43, 47, 22, 0, 31, 34, 28, 32, 46, 39, 11, 42, 1, 7, 30, 21, 25, 23, 44,
                         14],
                        [26, 16, 48, 29, 37, 15, 45, 33, 7, 35, 32, 23, 38, 39, 34, 31, 8, 10, 41, 49, 4, 28, 22, 25, 1,
                         18, 43, 17, 5, 36, 14, 11, 21, 27, 3, 20, 42, 44, 47, 40, 6, 30, 12, 2, 19, 24, 46, 0, 9, 13],
                        [0, 10, 18, 9, 48, 47, 24, 49, 19, 4, 28, 45, 27, 40, 34, 23, 29, 14, 37, 8, 38, 39, 16, 2, 5,
                         32, 33, 11, 41, 46, 13, 3, 35, 6, 44, 20, 22, 26, 12, 36, 1, 31, 30, 25, 42, 43, 21, 15, 17,
                         7],
                        [10, 32, 0, 43, 8, 13, 9, 35, 41, 30, 26, 20, 29, 48, 16, 33, 6, 24, 19, 14, 47, 38, 44, 5, 27,
                         18, 34, 42, 36, 4, 39, 49, 15, 7, 2, 23, 3, 21, 17, 40, 37, 45, 1, 22, 46, 31, 25, 28, 11, 12],
                        [5, 42, 22, 40, 13, 2, 20, 38, 30, 19, 1, 45, 34, 35, 32, 31, 15, 17, 26, 12, 28, 36, 25, 29,
                         41, 7, 8, 47, 49, 9, 4, 46, 23, 44, 37, 24, 14, 43, 10, 0, 3, 18, 16, 48, 11, 39, 21, 33, 6,
                         27],
                        [12, 13, 16, 49, 34, 48, 14, 25, 47, 15, 35, 21, 3, 22, 24, 18, 5, 28, 17, 40, 36, 31, 44, 42,
                         39, 32, 46, 45, 26, 2, 19, 6, 37, 30, 1, 29, 4, 41, 11, 8, 0, 27, 23, 43, 38, 7, 9, 20, 10,
                         33]])

    # Parameter Initialization Section
    if not len(x_ce):
        b = min(100, n_nu)
        # idx = [31, 39, 21, 33, 34, 5, 2, 15, 10, 29, 44, 32, 6, 37, 41, 27, 16, 40, 46, 13, 45, 7, 4, 47, 28, 20, 24, 36, 30, 48, 26, 49, 25, 42, 18, 43, 14, 0, 35, 22, 1, 3, 17, 23, 38, 12, 8, 19, 9, 11]
        idx = np.random.permutation(range(n_nu))
        x_ce = x_nu[:, idx[0:b]]
    # construct gaussian centers
    [_, n_ce] = shape(x_ce)

    sigma_list_rulsif = sigma_list(x_nu, x_de)

    # get lambda candidates
    lambda_list_rulsif = lambda_list()
    dist2_de = comp_dist(x_de, x_ce)
    # n_de * n_ce
    dist2_nu = comp_dist(x_nu, x_ce)
    # n_nu * n_ce

    # The Cross validation Section Begins
    lengthSigma = len(sigma_list_rulsif)
    lengthLamda = len(lambda_list_rulsif)
    score = zeros((len(sigma_list_rulsif), len(lambda_list_rulsif)))
    for i in range(0, len(sigma_list_rulsif)):
        k_de = kernel_gau(dist2_de, sigma_list_rulsif[i])
        k_nu = kernel_gau(dist2_nu, sigma_list_rulsif[i])
        for j in range(0, len(lambda_list_rulsif)):

            cv_index_nu = np.random.permutation(n_nu)
            # cv_index_nu = nu_perm[nu_index]
            # nu_index += 1
            temp1 = np.array(range(0, n_nu)) * fold
            temp1 = temp1 / n_nu
            cv_split_nu = np.floor(temp1) + 1
            cv_index_de = np.random.permutation(n_de)
            # cv_index_de = de_perm[de_index]
            # de_index += 1
            temp1 = np.array(range(0, n_de)) * fold
            temp1 = temp1 / n_de
            cv_split_de = np.floor(temp1) + 1

            loop_sum = 0
            for k in range(1, fold + 1):
                temp = (cv_split_de != k) * 1
                cv_index_de_removed = np.delete(cv_index_de, np.where(temp < 1))
                k_de_k = np.take(k_de, cv_index_de_removed, axis=0).conj().transpose()
                # n_ce * n_de
                temp = (cv_split_nu != k) * 1
                cv_index_nu_removed = np.delete(cv_index_nu, np.where(temp < 1))
                k_nu_k = np.take(k_nu, cv_index_nu_removed, axis=0).conj().transpose()
                # k_nu_k = k_nu[cv_index_nu[cv_split_nu != k], :].conj().transpose()
                # n_ce * n_nu
                k_de_k_trans = k_de_k.conj().transpose()
                k_nu_k_trans = k_nu_k.conj().transpose()
                # k_nu_k_size = size(k_nu_k, 2)
                [_, k_nu_k_size] = shape(k_nu_k)
                H_k = ((1 - alpha_rulsif) / k_nu_k_size) * k_de_k @ k_de_k_trans + (
                        alpha_rulsif / k_nu_k_size) * k_nu_k @ k_nu_k_trans
                h_k = np.mean(k_nu_k, 1)

                temp = numpy.identity(n_ce) * lambda_list_rulsif[j]
                temp = H_k + temp
                theta = linalg.solve(temp, h_k)
                # theta = max(theta, 0)

                # k_de_test = k_de[cv_index_de(cv_split_de == k), :].conj().transpose()
                temp = (cv_split_de == k) * 1
                cv_index_de_removed = np.delete(cv_index_de, np.where(temp < 1))
                k_de_test = np.take(k_de, cv_index_de_removed, axis=0).conj().transpose()
                # k_nu_test = k_nu[cv_index_nu(cv_split_nu == k), :].conj().transpose()
                temp = (cv_split_nu == k) * 1
                cv_index_nu_removed = np.delete(cv_index_nu, np.where(temp < 1))
                k_nu_test = np.take(k_nu, cv_index_nu_removed, axis=0).conj().transpose()
                # objective function value
                J = alpha_rulsif / 2 * mean((theta.conj().transpose() @ k_nu_test) ** 2) + (
                        1 - alpha_rulsif) / 2 * mean((theta.conj().transpose() @ k_de_test) ** 2) - mean(
                    theta.conj().transpose() @ k_nu_test)
                loop_sum = loop_sum + J
            score[i, j] = loop_sum / fold

    # find the chosen sigma and lambda
    [i_min, j_min] = np.where(score == np.min(score))
    sigma_chosen = sigma_list_rulsif[i_min]
    lambda_chosen = lambda_list_rulsif[j_min]

    # compute the final result
    k_de = kernel_gau(dist2_de.conj().transpose(), sigma_chosen)
    k_nu = kernel_gau(dist2_nu.conj().transpose(), sigma_chosen)

    H = ((1 - alpha_rulsif) / n_de) * k_de @ k_de.conj().transpose() + (
            alpha_rulsif / n_nu) * k_nu @ k_nu.conj().transpose()
    h = np.mean(k_nu, 1)
    theta = linalg.solve((H + eye(n_ce) * lambda_chosen), h)

    g_nu = theta.conj().transpose() @ k_nu
    g_de = theta.conj().transpose() @ k_de
    rPE = mean(g_nu) - 1 / 2 * (alpha_rulsif * mean(g_nu ** 2) + (1 - alpha_rulsif) * mean(g_de ** 2)) - 1 / 2

    return rPE, sigma_chosen, lambda_chosen


def sigma_list(x_nu, x_de):
    x = c_[x_nu, x_de]
    med = compmedDist(x.T)
    return med * array([0.6, 0.8, 1, 1.2, 1.4])


def lambda_list():
    return 10.0 ** array([-3, -2, -1, 0, 1])


def comp_dist(x, y):
    [d, nx] = shape(x)
    [d, ny] = shape(y)

    G = sum(np.multiply(x.conj().transpose(), x.conj().transpose()), 1)
    T = numpy.matlib.repmat(G, ny, 1)
    G = sum(np.multiply(y.conj().transpose(), y.conj().transpose()), 1)
    R = numpy.matlib.repmat(G, nx, 1)

    Ttrans = T.conj().transpose()
    xtrans = x.conj().transpose()
    xymulti = xtrans @ y
    dist2 = Ttrans + R - 2. * xymulti
    return dist2


def norm_pdf(x, mu, std):
    return exp(-(x - mu) ** 2 / (2 * (std ** 2))) / (std * sqrt(2 * pi))


def change_detection(X, n, k, alpha, fold):
    WIN = sliding_window(X, k, 1)
    nSamples = size(WIN, 1)

    t = n + 1
    sigma_track = []
    lambda_track = []
    index = 0

    SCORE = np.array(zeros(nSamples - (t + n - 2)))

    while t + n - 2 < nSamples:

        stuff = [WIN[:, (t - n) - 1: (n + t) - 1]]

        Y = np.array(stuff[0])
        #        Y = Y / np.matlib.repmat(std(Y, 0, 2), 1, 2 * n)
        ySTD = std(Y, 1, ddof=1)
        yStdTrans = ySTD.conj().transpose()
        RepMat = np.matlib.repmat(yStdTrans, 1, 2 * n)
        RepMatReShape = np.reshape(RepMat, (n*2, k))
        yRepMat = RepMatReShape.conj().transpose()
        Y = np.true_divide(Y, yRepMat)
        Ylen = shape(Y)
        Ylength = Ylen[1]
        YRef = Y[:, 0:n]
        YTest = Y[:, n:Ylength]
        #        (PE, w, s) = R_ULSIF(x_nu, x_de, c_[x_re, x_nu, x_de], alpha,
        #                             sigma_list(x_nu, x_de), lambda_list(), x_nu.shape[1], 5)
        [_, n_nu] = shape(YTest)
        b = min(100, n_nu)
        [s, sig, lam] = R_ULSIF(YRef, YTest, [], [], alpha, fold)
        # (x_nu, x_de, x_re, alpha, sigma_list, lambda_list, b, fold)
        # [s, _, sig, lam] = R_ULSIF_PY(YRef, YTest, np.array([]), alpha, sigma_list(YRef, YTest), lambda_list(), b, fold)
        sigma_track = [sigma_track, sig]
        lambda_track = [lambda_track, lam]

        # print out the progress
        if mod(t, 20) == 0:
            print(t)

        SCORE[index] = s
        t = t + 1
        index += 1

    return SCORE, sigma_track, lambda_track


def sliding_window(X, windowSize, step):
    offset = windowSize * step;
    xShape = X.shape
    num_dims = xShape[0]
    num_samples = xShape[1]
    m = num_dims * windowSize * step
    n = int(floor(num_samples / windowSize / step))
    WINDOWS = zeros((m, num_samples - (offset - 1)))

    for i in range(0, num_samples, step):
        if i + offset > num_samples:
            break
        w = X[:, i: i + offset].conj().transpose()
        WINDOWS[:, int(ceil(i / step))] = w.ravel()

    return WINDOWS


# Reversing a list using reversed()
def Reverse(lst):
    return [ele for ele in reversed(lst)]


seed(1)
mat_contents = sio.loadmat('logwell.mat')
y = mat_contents['y']
y_trans = mat_contents['y'].conj().transpose()
y_shape = shape(y_trans)
y_axis = y_shape[0]
alpha = .01;

n = 50
k = 10
subplot(2, 1, 1)
plot(y_trans)
pr = cProfile.Profile()
pr.enable()
[score1, _, _] = change_detection(y, n, k, alpha, 5)
pr.disable()
# after your program ends
pr.print_stats(sort="calls")
[score2, _, _] = change_detection(np.array(Reverse(y)), n, k, alpha, 5)

subplot(2, 1, 2)
final_score = score1 + score2
numpy.savetxt("score1.csv", score1, delimiter=",")
numpy.savetxt("score2.csv", score2, delimiter=",")
numpy.savetxt("final_score.csv", final_score, delimiter=",")
zeros_score = zeros(2 * n - 2 + k)
plot_array = np.concatenate((zeros_score, final_score))
plot(plot_array)
# axis([0, y_axis, 0, 100])
title('Change-Point Score')

# plot(y_trans, label='r_{hat}(x)', linewidth=2.5, color='red')
show()
