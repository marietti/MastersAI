from os.path import dirname, join as pjoin
import scipy.io as sio
from pylab import *
from scipy import linalg
import numpy.matlib
import scipy
from scipy import spatial
import math

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


def kernel_Gaussian(x, c, sigma):
    (d, nx) = x.shape
    (d, nc) = c.shape
    x2 = sum(x ** 2, 0)
    c2 = sum(c ** 2, 0)

    distance2 = tile(c2, (nx, 1)) + \
                tile(x2[:, newaxis], (1, nc)) \
                - 2 * dot(x.T, c)

    return exp(-distance2 / (2 * (sigma ** 2)));


def R_ULSIF_PY(x_nu, x_de, x_re, alpha, b, fold):
    # x_nu: samples from numerator
    # x_de: samples from denominator
    # x_re: reference sample
    # alpha: alpha defined in relative density ratio
    # sigma_list, lambda_list: parameters for model selection
    # b: number of kernel basis
    # fold: number of fold for cross validation

    sigma_list_rulsif = sigma_list(x_nu, x_de)
    lambda_list_rulsif = lambda_list()

    (d, n_nu) = x_nu.shape
    (d, n_de) = x_de.shape
    rand_index = permutation(n_nu)
    b = min(b, n_nu)
    # x_ce = x_nu[:,rand_index[0:b]]
    x_ce = x_nu[:, r_[0:b]]

    score_cv = zeros((size(sigma_list_rulsif),
                      size(lambda_list_rulsif)))

    cv_index_nu = permutation(n_nu)
    # cv_index_nu = r_[0:n_nu]
    cv_split_nu = floor(r_[0:n_nu] * fold / n_nu)
    cv_index_de = permutation(n_de)
    # cv_index_de = r_[0:n_de]
    cv_split_de = floor(r_[0:n_de] * fold / n_de)

    for sigma_index in r_[0:size(sigma_list_rulsif)]:
        sigma = sigma_list_rulsif[sigma_index]
        K_de = kernel_Gaussian(x_de, x_ce, sigma).T
        K_nu = kernel_Gaussian(x_nu, x_ce, sigma).T

        score_tmp = zeros((fold, size(lambda_list_rulsif)))

        for k in r_[0:fold]:
            Ktmp1 = K_de[:, cv_index_de[cv_split_de != k]]
            Ktmp2 = K_nu[:, cv_index_nu[cv_split_nu != k]]

            Ktmp = alpha / Ktmp2.shape[1] * dot(Ktmp2, Ktmp2.T) + \
                   (1 - alpha) / Ktmp1.shape[1] * dot(Ktmp1, Ktmp1.T)

            mKtmp = mean(K_nu[:, cv_index_nu[cv_split_nu != k]], 1)

            for lambda_index in r_[0:size(lambda_list_rulsif)]:
                lbd = lambda_list_rulsif[lambda_index]

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

    lambda_chosen = lambda_list_rulsif[lambda_chosen_index[sigma_chosen_index]]
    sigma_chosen = sigma_list_rulsif[sigma_chosen_index]

    K_de = kernel_Gaussian(x_de, x_ce, sigma_chosen).T
    K_nu = kernel_Gaussian(x_nu, x_ce, sigma_chosen).T

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

    PE = mean(wh_x_nu) - 1. / 2 * (alpha * mean(wh_x_nu ** 2) +
                                   (1 - alpha) * mean(wh_x_de ** 2)) - 1. / 2

    return PE, score, lambda_chosen, sigma_chosen

def kernel_gau(dist2, sigma):
    k = exp(-dist2/(2*sigma**2))
    return k

def R_ULSIF (x_de, x_nu, x_re, x_ce, alpha_rulsif = 0.5, fold=5):

    [_,n_nu] = shape(x_nu)
    [_,n_de] = shape(x_de)

    # Parameter Initialization Section
    if not len(x_ce):
        b = min(100, n_nu)
        idx = np.random.permutation(n_nu)
        x_ce = x_nu[:, idx[0: b]]
    # construct gaussian centers
    [_,n_ce] = shape(x_ce)

    sigma_list_rulsif = sigma_list(x_nu, x_de)

    # get lambda candidates
    lambda_list_rulsif = lambda_list()
    x_nu_reshape = np.reshape(x_de, (50, 10))
    x_de_reshape = np.reshape(x_de, (50, 10))
    x_ce_reshape = np.reshape(x_ce, (50, 10))
    dist2_de = scipy.spatial.distance.cdist(x_de_reshape, x_ce_reshape)
    # n_de * n_ce
    dist2_nu = scipy.spatial.distance.cdist(x_nu_reshape, x_ce_reshape)
    # n_nu * n_ce

    # The Cross validation Section Begins
    lengthSigma = len(sigma_list_rulsif)
    lengthLamda = len(lambda_list_rulsif)
    score = zeros((len(sigma_list_rulsif), len(lambda_list_rulsif)))
    for i in range(1, len(sigma_list_rulsif)):
        k_de = kernel_gau(dist2_de, sigma_list_rulsif[i])
        k_nu = kernel_gau(dist2_nu, sigma_list_rulsif[i])
        for j in range(1, len(lambda_list_rulsif)):

            cv_index_nu = np.random.permutation(n_nu)
            temp1 = n_nu
            cv_split_nu = math.floor(temp1*fold/n_nu)+1
            cv_index_de = np.random.permutation(n_de)
            temp1 = n_de
            cv_split_de = math.floor(temp1*fold/n_de)+1

            loop_sum = 0
            for k in range(1, fold):
                k_de_k = k_de[cv_index_de[cv_split_de!=k],:].conj().transpose()
                # n_ce * n_de
                k_nu_k = k_nu[cv_index_nu[cv_split_nu!=k],:].conj().transpose()
                # n_ce * n_nu
                k_de_k_trans = k_de_k.conj().transpose()
                k_nu_k_trans = k_nu_k.conj().transpose()
                k_nu_k_size = size(k_nu_k, 2)
                H_k = ((1 - alpha_rulsif) / size(k_de_k, 2)) * k_de_k * k_de_k_trans + (alpha_rulsif / k_nu_k_size) * k_nu_k * k_nu_k_trans
                h_k = mean(k_nu_k, 2)

                theta = (H_k + numpy.identity(n_ce) * lambda_list_rulsif[j])/h_k
                # theta = max(theta, 0)

                k_de_test = k_de[cv_index_de(cv_split_de == k),:].conj().transpose()
                k_nu_test = k_nu[cv_index_nu(cv_split_nu == k),:].conj().transpose()
                # objective function value
                J = alpha_rulsif / 2 * mean((theta.conj().transpose() * k_nu_test) ^ 2) + (1 - alpha_rulsif) / 2 * mean((theta.conj().transpose() * k_de_test) ^ 2) - mean(theta.conj().transpose() * k_nu_test)
                loop_sum = loop_sum + J
        score[i, j] = loop_sum / fold

    # find the chosen sigma and lambda
    [i_min, j_min] =  np.where(score == np.min(score))
    sigma_chosen = sigma_list_rulsif(i_min)
    lambda_chosen = lambda_list_rulsif(j_min)

    # compute the final result
    k_de = kernel_gau(dist2_de.conj().transpose(), sigma_chosen)
    k_nu = kernel_gau(dist2_nu.conj().transpose(), sigma_chosen)

    H = ((1 - alpha_rulsif) / n_de) * k_de * k_de.conj().transpose() + (alpha_rulsif / n_nu) * k_nu * k_nu.conj().transpose()
    h = mean(k_nu, 2)

    theta = (H + eye(n_ce) * lambda_chosen)/h

    g_nu = theta.conj().transpose() *k_nu
    g_de = theta.conj().transpose() *k_de
    g_re = []
    if not x_re.size:
        dist2_re = comp_dist(x_re, x_ce)
        k_re = kernel_gau(dist2_re.conj().transpose(), sigma_chosen)
        g_re = theta.conj().transpose() * k_re
    rPE = mean(g_nu) - 1 / 2 * (alpha_rulsif * mean(g_nu ^ 2) + (1 - alpha_rulsif) * mean(g_de ^ 2)) - 1 / 2

    return rPE, g_nu, g_re, sigma_chosen, lambda_chosen

def sigma_list(x_nu, x_de):
    x = c_[x_nu, x_de]
    med = compmedDist(x.T)
    return med * array([0.6, 0.8, 1, 1.2, 1.4])


def lambda_list():
    return 10.0 ** array([-3, -2, -1, 0, 1])

def comp_dist(x, y):
    [d, nx] = size(x)
    [d, ny] = size(y)

    G = sum(x*x,1)
    T = numpy.matlib.repmat(G,ny,1)
    G = sum(y*y,1)
    R = numpy.matlib.repmat(G,nx,1)

    dist2 = T.conj().transpose()  + R - 2.*x.conj().transpose() *y;
    return dist2

def norm_pdf(x, mu, std):
    return exp(-(x - mu) ** 2 / (2 * (std ** 2))) / (std * sqrt(2 * pi))


def change_detection(X, n, k, alpha, fold):

    SCORE = []

    WIN = sliding_window(X, k, 1)
    nSamples = size(WIN, 1)
    t = n + 1
    sigma_track = []
    lambda_track = []
    index = 0

    lambda_list_rulsif = lambda_list()
    while t + n - 1 <= nSamples-1:

        stuff = [WIN[:, t - n: n + t]]

        Y = np.array(stuff[0])
#        Y = Y / np.matlib.repmat(std(Y, 0, 2), 1, 2 * n)
        ySTD = std(Y, 1)
        yRepMat = np.matlib.repmat(ySTD, 1, 2 * n)
        Y = np.true_divide(Y, np.reshape(yRepMat, (10, 100)))
        Ylen = shape(Y)
        Ylength = Ylen[1]
        YRef = Y[:, 0:n]
        YTest = Y[:, n:Ylength]
#        (PE, w, s) = R_ULSIF(x_nu, x_de, c_[x_re, x_nu, x_de], alpha,
#                             sigma_list(x_nu, x_de), lambda_list(), x_nu.shape[1], 5)
        [_, n_nu] = shape(YTest)
        b = min(100, n_nu)
        [s, _, _, sig, lam] = R_ULSIF(YRef, YTest, [], [], alpha, fold)
#        [s, _, sig, lam] = R_ULSIF_PY(YRef, YTest, np.array([]), alpha, b, fold)
        sigma_track = [sigma_track, sig]
        lambda_track = [lambda_track, lam]

        # print out the progress
        if mod(t, 20) == 0:
            print(t)

        SCORE = [SCORE, s]
        t = t + 1
        index += 1

    return SCORE, sigma_track, lambda_track


def sliding_window(X, windowSize, step):

    offset = windowSize * step;
    xShape = X.shape
    num_dims = xShape[0]
    num_samples = xShape[1]
    m = num_dims * windowSize * step
    n = int(floor(num_samples/windowSize/step))
    WINDOWS = zeros((m, num_samples - (offset - 1)))

    for i in range(step, (num_samples - (offset - 1))):
        w = X[:, i: i + offset].conj().transpose()
        WINDOWS[:, int(ceil(i/step))] = w.ravel()

    return WINDOWS

# Reversing a list using reversed()
def Reverse(lst):
    return [ele for ele in reversed(lst)]

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
[score1, _, _] = change_detection(y, n, k, alpha, 5)
[score2, _, _] = change_detection(np.array(Reverse(y)), n, k, alpha, 5)

subplot(2, 1, 2)
score2 = Reverse(score2)
change = score1 + score2
plot(np.array(change))
#axis([0, y_axis, 0, 100])
title('Change-Point Score')

#plot(y_trans, label='r_{hat}(x)', linewidth=2.5, color='red')
show()