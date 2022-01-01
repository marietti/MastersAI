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

    # Parameter Initialization Section
    if not len(x_ce):
        b = min(100, n_nu)
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
    score = zeros((len(sigma_list_rulsif), len(lambda_list_rulsif)))
    for i in range(0, len(sigma_list_rulsif)):
        k_de = kernel_gau(dist2_de, sigma_list_rulsif[i])
        k_nu = kernel_gau(dist2_nu, sigma_list_rulsif[i])
        for j in range(0, len(lambda_list_rulsif)):

            cv_index_nu = np.random.permutation(n_nu)
            temp1 = np.array(range(0, n_nu)) * fold
            temp1 = temp1 / n_nu
            cv_split_nu = np.floor(temp1) + 1
            cv_index_de = np.random.permutation(n_de)
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
        RepMatReShape = np.reshape(RepMat, (100, k))
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

def change_detection_single(X, n, k, alpha, fold):
    WIN = sliding_window(X, k, 1)
    Y = np.array(WIN)
    ySTD = std(Y, 1, ddof=1)
    yStdTrans = ySTD.conj().transpose()
    RepMat = np.matlib.repmat(yStdTrans, 1, 2 * n)
    sizeRepMat = size(RepMat)
    RepMatReShape = np.reshape(RepMat, (int(sizeRepMat/k), k))
    yRepMat = RepMatReShape.conj().transpose()
    Y = np.true_divide(Y, yRepMat)
    Ylen = shape(Y)
    Ylength = Ylen[1]
    YRef = Y[:, 0:n]
    YTest = Y[:, n:Ylength]
    [s, sig, lam] = R_ULSIF(YRef, YTest, [], [], alpha, fold)

    return s, sig, lam

def sliding_window(X, windowSize, step):
    offset = windowSize * step
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
