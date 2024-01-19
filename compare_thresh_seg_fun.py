import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import random
import scipy.linalg as la
from sklearn.metrics import confusion_matrix
import time
from scipy.stats import gamma
from config import win,level,L,factor0, min_factor1,min_factor2,nWindow,thres_type,pfa

def preprocessSAR(input_data):
    Pv=input_data
    Pd=input_data
    data = input_data
    Para1 = Pv
    Para2 = Pd
    mask_Pd_neg = Para2 < 0
    Para2[mask_Pd_neg] = 0
    mask_Pd_pos = Para2 > np.max(data)
    Para2[mask_Pd_pos] = np.max(data.real)
    img1 = f_SAR_visualize_yip(Para1, 'meanshift', 1, None)
    img1 = np.round(255 * img1.real).astype(np.uint8)
    img2 = f_SAR_visualize_yip(Para2, 'meanshift', 1, None)
    img2 = np.round(255 * img2.real).astype(np.uint8)
    return img1, img2
def preprocess(input_data):
    input_data = input_data['real'] + input_data['imag'] * 1j

    temp = np.ones((win, win)) / win / win
    output_data = np.zeros_like(input_data)
    for channel in range(3):
        output_data[:, :, channel, channel] = convolve2d(input_data[:, :, channel, channel], temp, mode='same',
                                                         boundary='symm')
    Ps, Pd, Pv = freeman_decom_T(output_data)
    data = input_data[:, :, 0, 0] + input_data[:, :, 1, 1] + input_data[:, :, 2, 2]
    Para1 = Pv
    Para2 = Pd
    mask_Pd_neg = Para2 < 0
    Para2[mask_Pd_neg] = 0
    mask_Pd_pos = Para2 > np.max(data)
    Para2[mask_Pd_pos] = np.max(data.real)
    img1 = f_SAR_visualize_yip(Para1, 'meanshift', 1, None)
    img1 = np.round(255 * img1.real).astype(np.uint8)
    img2 = f_SAR_visualize_yip(Para2, 'meanshift', 1, None)
    img2 = np.round(255 * img2.real).astype(np.uint8)
    return img1, img2


def var_parallel(image, nWindow):
    rwin = (nWindow - 1) // 2
    siz = image.shape
    h_limit = siz[0]
    v_limit = siz[1]
    image_avg = np.zeros(siz)
    image_var = np.zeros((siz[0], siz[1]))
    if len(siz) == 3:
        image = np.pad(image, ((rwin, rwin), (rwin, rwin), (0, 0)), mode='edge')
    else:
        image = np.pad(image, ((rwin, rwin), (rwin, rwin)), mode='edge')
    for i in range(-rwin, rwin + 1):
        for j in range(-rwin, rwin + 1):
            if len(siz) == 3:
                image_avg += image[rwin + i:h_limit + rwin + i, rwin + j:v_limit + rwin + j, :]
            else:
                image_avg += image[rwin + i:h_limit + rwin + i, rwin + j:v_limit + rwin + j]
    image_avg = image_avg / (nWindow * nWindow)
    for i in range(-rwin, rwin + 1):
        for j in range(-rwin, rwin + 1):
            if len(siz) == 3:
                image_var += image[rwin + i:h_limit + rwin + i, rwin + j:v_limit + rwin + j] ** 2
            else:
                image_var += image[rwin + i:h_limit + rwin + i, rwin + j:v_limit + rwin + j] ** 2
    image_var = image_var / (nWindow * nWindow) - image_avg ** 2
    return image_avg, image_var


def f_sort_eq_new(img, N):
    temp = np.sort(img.flatten())
    gf = np.zeros(N)
    gf[0:(N - 1)] = temp[(np.floor(np.arange(1, N) * len(temp) / N)).astype(int)]
    gf[N - 1] = temp[-1]
    Res_img = np.zeros_like(img)
    for i in range(1, N - 1):
        Res_img += (i - 1) * ((img <= gf[i]) & (img > gf[i - 1]))
    Res_img += (N - 1) * (img > gf[N - 2])
    Res_img = Res_img / (N - 1)
    return Res_img


def fPauliImShow_new(data):
    siz = data.shape
    z = np.zeros((siz[0], siz[1], 3))
    if siz[0] == 3:
        z[:, :, 2] = f_sort_eq_new(data[0, 0, :, :], level)
        z[:, :, 0] = f_sort_eq_new(data[1, 1, :, :], level)
        z[:, :, 1] = f_sort_eq_new(data[2, 2, :, :], level)
    elif siz[2] == 3:
        z[:, :, 2] = f_sort_eq_new(data[:, :, 0, 0], level)
        z[:, :, 0] = f_sort_eq_new(data[:, :, 1, 1], level)
        z[:, :, 1] = f_sort_eq_new(data[:, :, 2, 2], level)
    elif siz[2] == 9:
        z[:, :, 2] = f_sort_eq_new(data[:, :, 0], level)
        z[:, :, 0] = f_sort_eq_new(data[:, :, 1], level)
        z[:, :, 1] = f_sort_eq_new(data[:, :, 2], level)
    else:
        print('error')
        return
    return z


def freeman_decom_T(T):
    xdim, ydim, _, _ = T.shape

    T11 = T[:, :, 0, 0]
    T12 = T[:, :, 0, 1]
    T22 = T[:, :, 1, 1]
    T33 = T[:, :, 2, 2]

    Ps = np.zeros((xdim, ydim))
    Pd = np.zeros((xdim, ydim))
    Pv = np.zeros((xdim, ydim))

    Pv = 4 * T33
    a1 = T11 - 2 * T33
    a2 = T22 - T33

    flag1 = np.where((a1 - a2) > 0)
    flag2 = np.where((a1 - a2) <= 0)
    flag3 = np.where(a1 == 0)
    flag4 = np.where(a2 == 0)

    Pd[flag3] = a2[flag3].real
    Ps[flag3] = 0

    Ps[flag4] = a1[flag4].real
    Pd[flag4] = 0

    Ps[flag1] = a1[flag1].real + (np.abs(T12[flag1]) ** 2) / a1[flag1].real
    Pd[flag1] = a2[flag1].real - (np.abs(T12[flag1]) ** 2) / a1[flag1].real

    Pd[flag2] = a2[flag2].real + (np.abs(T12[flag2]) ** 2) / a2[flag2].real
    Ps[flag2] = a1[flag2].real - (np.abs(T12[flag2]) ** 2) / a2[flag2].real

    return Ps, Pd, Pv


def f_meanshift(X, h, initialvalue=None):
    x = -1
    if initialvalue != None:
        y = initialvalue
    else:
        y = np.mean(X)
    count = 0
    while abs(x - y) > 0.0001 and count < 100000:
        x = y
        numerator = np.sum(X * np.exp(-(X - x) ** 2 / (2 * h ** 2)))
        denominator = np.sum(np.exp(-(X - x) ** 2 / (2 * h ** 2)))
        y = numerator / denominator
        count += 1
    return y


def f_SAR_visualize_yip(img, imgtype, L, modify=0):
    img[np.isnan(img)] = 0
    Vector_img = img.ravel()
    Vector_img = Vector_img[Vector_img != 0]
    IM = np.zeros(img.shape)
    if imgtype == 'meanshift':
        y = f_meanshift(Vector_img, 5)
        k = np.log(2) / y
        IM = 1 - np.exp(-k * img)
    IM = np.round(IM * 255) / 255
    return IM


# OTSU
def lc_otsu(x):

    pdf = np.histogram(x, bins=range(257), density=True)[0]
    uT = np.sum(np.arange(256) * pdf)
    var_max = 0
    k_max = 0
    for k in range(L):
        w0 = np.sum(pdf[:k + 1])
        if 0 < w0 < 1:
            uk = np.sum(np.arange(k + 1) * pdf[:k + 1])
            vark = (uT * w0 - uk) ** 2 / (w0 * (1 - w0))
            if vark > var_max:
                var_max = vark
                k_max = k
    y = x <= k_max
    return y


def otsu(img1, img2):
    xsize = img1.shape[0]
    ysize = img1.shape[1]
    Pv_OSTU = lc_otsu(img1)
    Pd_OSTU = lc_otsu(img2)
    Pd_OSTU = ~Pd_OSTU & ~Pv_OSTU
    X0 = 2 * np.ones((xsize, ysize), dtype=np.uint8)
    X0[Pv_OSTU] = 1
    X0[Pd_OSTU] = 3
    return X0


# 最大熵
def my_entropy(pdf):
    L = len(pdf)
    H = 0
    for k in range(L):
        if pdf[k] > 0 and pdf[k] < 1:
            H = H - pdf[k] * np.log(pdf[k])
    return H


def lc_entropy(x, L):
    M, N = x.shape
    pdf = np.histogram(x, bins=L, range=(0, L - 1))[0] / (M * N)
    pdf = pdf.T
    Hn = my_entropy(pdf[:L])
    H_max = 0
    for k in range(L - 1):
        ps = np.sum(pdf[:k + 1])
        if ps > 0 and ps < 1:
            Hs = my_entropy(pdf[:k + 1])
            H = np.log(ps * (1 - ps)) + Hs + (Hn - Hs) / (1 - ps)
            if H > H_max:
                H_max = H
                k_max = k
    y = x <= k_max
    return y


def entropy(img1, img2):
    xsize = img1.shape[0]
    ysize = img1.shape[1]
    Pv_entropy = lc_entropy(img1, 256)
    Pd_entropy = lc_entropy(img2, 256)
    X0 = 2 * np.ones((xsize, ysize), dtype=np.uint8)
    X0[Pv_entropy] = 3
    X0[Pd_entropy] = 1
    return X0


# 迭代法
def lc_iterative(x):
    dimx, dimy = x.shape
    x_max = np.max(x)
    x_min = np.min(x)
    T = random.uniform(x_min, x_max)
    while True:
        m1 = np.mean(x[x > T])
        m2 = np.mean(x[x <= T])
        T1 = (m1 + m2) / 2
        if T1 == T:
            break
        T = T1
    y = np.zeros((dimx, dimy))
    y[x <= T] = 1
    return y


def iterative(img1, img2):
    xsize = img1.shape[0]
    ysize = img1.shape[1]
    Pv_iterative = lc_iterative(img1)
    Pd_iterative = lc_iterative(img2)
    Pd_iterative = np.logical_not(Pd_iterative) & np.logical_not(Pv_iterative)
    X0 = 2 * np.ones((xsize, ysize), dtype=np.uint8)
    X0[Pv_iterative == 1] = 1
    X0[Pd_iterative == 1] = 3
    return X0


# 矩方法
def lc_moment(x):
    M, N = x.shape
    pdf = np.histogram(x, bins=256, range=(0, 256))[0] / (M * N)
    pdf = pdf.reshape((1, -1))
    m0 = 0
    m1 = 0
    m2 = 0
    m3 = 0
    for i in range(256):
        m0 += pdf[0, i]
        m1 += i * pdf[0, i]
        m2 += (i ** 2) * pdf[0, i]
        m3 += (i ** 3) * pdf[0, i]

    cd = la.det(np.array([[m0, m1], [m1, m2]]))
    c0 = la.det(np.array([[-m2, m1], [-m3, m2]])) / cd
    c1 = la.det(np.array([[m0, -m2], [m1, -m3]])) / cd
    z0 = (-c1 - np.sqrt(c1 ** 2 - 4 * c0)) / 2
    z1 = (-c1 + np.sqrt(c1 ** 2 - 4 * c0)) / 2
    pd = la.det(np.array([[1, 1], [z0, z1]]))
    p0 = la.det(np.array([[1, 1], [m1, z1]])) / pd
    p00 = 0
    err_min = 1
    T = 0
    for k in range(1, 257):
        p00 += pdf[0, k - 1]
        err = abs(p00 - p0)
        if err <= err_min:
            err_min = err
            T = k - 1
    y = np.zeros((M, N), dtype=np.uint8)
    y[x <= T] = 1
    return y


def moment(img1, img2):
    xsize = img1.shape[0]
    ysize = img1.shape[1]
    Pv_moment = lc_moment(img1)
    Pd_moment = lc_moment(img2)
    Pd_moment = np.logical_not(Pd_moment) & np.logical_not(Pv_moment)
    X0 = 2 * np.ones((xsize, ysize), dtype=np.uint8)
    X0[Pv_moment == 1] = 1
    X0[Pd_moment == 1] = 3
    return X0


# EM
def distribution(m, v, g, x):
    x = x.flatten()
    m = m.flatten()
    v = v.flatten()
    g = g.flatten()
    y = np.zeros((len(x), len(m)))
    for i in range(len(m)):
        d = x - m[i]
        amp = g[i] / np.sqrt(2 * np.pi * v[i])
        y[:, i] = amp * np.exp(-0.5 * (d * d) / v[i])
    return y


def histogram(datos):
    datos = datos.flatten()
    datos = np.nan_to_num(datos)
    tam = len(datos)
    m = int(np.ceil(np.max(datos)) + 1)
    h = np.zeros(m)
    for i in range(tam):
        f = int(np.floor(datos[i]))
        if f > 0 and f < (m - 1):
            a2 = datos[i] - f
            a1 = 1 - a2
            h[f] += a1
            h[f + 1] += a2
    h = np.convolve(h, [1, 2, 3, 2, 1], 'same')
    h = h[2:(len(h) - 2)]
    h = h / np.sum(h)
    return h


def EMSeg(ima, k):
    ima = ima.astype(float)
    copy = np.copy(ima)
    ima = ima.flatten()
    mi = np.min(ima)
    ima = ima - mi + 1
    m = np.max(ima)
    s = len(ima)
    h, x = np.histogram(ima, bins=np.arange(1, m + 2))
    x = x[:-1]
    h = h.astype(float)
    mu = np.arange(1, k + 1) * m / (k + 1)
    v = np.ones(k) * m
    p = np.ones(k) / k
    sml = np.mean(np.diff(x)) / 1000
    while True:
        prb = distribution(mu, v, p, x)
        scal = np.sum(prb, axis=1) + np.finfo(float).eps
        loglik = np.sum(h * np.log(scal))

        for j in range(k):
            pp = h * prb[:, j] / scal
            p[j] = np.sum(pp)
            mu[j] = np.sum(x * pp) / p[j]
            vr = (x - mu[j])
            v[j] = np.sum(vr * vr * pp) / p[j] + sml
        p = p + 1e-3
        p = p / np.sum(p)
        prb = distribution(mu, v, p, x)
        scal = np.sum(prb, axis=1) + np.finfo(float).eps
        nloglik = np.sum(h * np.log(scal))
        if nloglik - loglik < 0.0001:
            break
    mu = mu + mi - 1
    s = copy.shape
    mask = np.zeros(s, dtype=int)
    for i in range(s[0]):
        for j in range(s[1]):
            c = np.zeros(k)
            for n in range(k):
                c[n] = distribution(mu[n], v[n], p[n], copy[i, j])
            a = np.argwhere(c == np.max(c))
            mask[i, j] = a[0, 0]
    return mask, mu, v, p


def EM(img1, img2):
    xsize = img1.shape[0]
    ysize = img1.shape[1]
    Pv_EM, mu, v, p = EMSeg(img1, 2)
    Pd_EM, mu, v, p = EMSeg(img2, 2)
    X0 = np.ones((xsize, ysize), dtype=np.uint8)
    X0[Pv_EM == 1] = 2
    X0[Pd_EM == 1] = 3
    return X0


# 指标
def metrics_all(X0, X1):
    cm = confusion_matrix(X1.flatten(), X0.flatten())
    nclass = len(np.unique(X1))
    acc = np.zeros(nclass)
    iou = np.zeros(nclass)
    total_acc = 0
    for k in range(nclass):
        acc[k] = cm[k, k] / np.sum(cm[:, k])
        iou[k] = cm[k, k] / (np.sum(cm[k, :]) + np.sum(cm[:, k]) - cm[k, k])
        total_acc += cm[k, k]

    total_acc = total_acc / np.sum(cm)
    miou = np.mean(iou)
    return acc, total_acc, iou, miou


def afterprocess(mat2):
    data2 = mat2.get('wishart_seg')
    data2 = np.array(data2)
    X1 = np.transpose(data2)
    where_1 = np.where(X1 == 1)
    where_3 = np.where(X1 == 3)
    X1[where_1] = 3
    X1[where_3] = 1
    return X1



def aver_single_one(T, seg_I):
    return np.mean(T[seg_I])


def Trace3_HM1xHM2(temp, image_vec):
    tr = np.zeros((image_vec.shape[0], image_vec.shape[1]))
    tr[:, :] = temp[0, 0] * image_vec[:, :, 0, 0] + temp[0, 1] * image_vec[:, :, 1, 0] + temp[0, 2] * image_vec[:, :, 2,
                                                                                                      0] + \
               temp[1, 0] * image_vec[:, :, 0, 1] + temp[1, 1] * image_vec[:, :, 1, 1] + temp[1, 2] * image_vec[:, :, 2,
                                                                                                      1] + \
               temp[2, 0] * image_vec[:, :, 0, 2] + temp[2, 1] * image_vec[:, :, 1, 2] + temp[2, 2] * image_vec[:, :, 2,
                                                                                                      2]
    return tr


def var_parallel2(image, nWindow):
    rwin = (nWindow - 1) // 2
    siz = image.shape
    h_limit = siz[0]
    v_limit = siz[1]
    image_avg = np.zeros(siz)
    image_var = np.zeros((siz[0], siz[1]))

    if len(siz) == 4:
        image = np.pad(image, ((rwin, rwin), (rwin, rwin), (0, 0), (0, 0)), mode='edge')
        image_temp1 = Trace3_HM1xHM2(image, image)
    else:
        image = np.pad(image, ((rwin, rwin), (rwin, rwin)), mode='edge')

    for i in range(-rwin, rwin + 1):
        for j in range(-rwin, rwin + 1):
            if len(siz) == 4:
                image_avg += image[rwin + i:h_limit + rwin + i, rwin + j:v_limit + rwin + j, :, :]
            else:
                image_avg += image[rwin + i:h_limit + rwin + i, rwin + j:v_limit + rwin + j]

    image_avg /= nWindow * nWindow

    for i in range(-rwin, rwin + 1):
        for j in range(-rwin, rwin + 1):
            if len(siz) == 4:
                image_var += image_temp1[rwin + i:h_limit + rwin + i, rwin + j:v_limit + rwin + j]
            else:
                image_var += image[rwin + i:h_limit + rwin + i, rwin + j:v_limit + rwin + j] ** 2

    if len(siz) == 4:
        image_avg = Trace3_HM1xHM2(image_avg, image_avg)
        image_var = image_var / nWindow / nWindow - image_avg
    else:
        image_var = image_var / nWindow / nWindow - image_avg ** 2
    return image_avg, image_var


def likelihood_ratio_single(Para1, thres):
    tmask = Para1 < thres
    mu1 = aver_single_one(Para1, tmask)
    n1 = np.sum(tmask)
    mu2 = aver_single_one(Para1, ~tmask)
    n2 = np.sum(~tmask)
    siz = Para1.shape
    mu = aver_single_one(Para1, np.ones(siz, dtype=bool))
    n = siz[0] * siz[1]
    pratio = -n1 * np.log(mu1) - n2 * np.log(mu2) + n * np.log(mu)
    return pratio


def para_estimation(data):
    data = np.ravel(data)
    len_data = len(data)
    sigma = np.mean(data)
    numerator = sigma ** 2
    denominator = np.sum((data - sigma) ** 2)
    L = numerator / denominator * (len_data - 1)
    return L, sigma

def gamma_thres(data, pfa):
    L, sigma = para_estimation(data)
    xxxx = np.arange(0, 10, 1e-6)
    pd_cdf1 = gamma.cdf(xxxx, L, scale=sigma / L)
    ind = np.argmin(np.abs(pd_cdf1 - 1 + pfa))
    thres = xxxx[ind]
    return thres


def pdf_fitting_single(npara1, mask):
    data_temp = npara1[mask]
    L, sigma = para_estimation(data_temp)
    fig, ax = plt.subplots()
    xx, yy = np.histogram(data_temp, bins=100)
    bin_width = (yy[1] - yy[0])
    xxx = np.arange(0, bin_width * 100, bin_width)
    yyy = gamma.pdf(xxx, L, scale=sigma / L)
    ax.plot(xxx, yyy, 'r.-')
    ax.bar(yy[:-1], xx / np.sum(mask) / (yy[1] - yy[0]), width=bin_width, color='w')
    ax.set_xlabel('intensity', fontsize=15)
    ax.set_ylabel('density', fontsize=15)
    plt.show()


def thres_deter(outputIm, mask_hom, percent):
    hist_detector, center = np.histogram(outputIm[mask_hom], bins=100)
    hist_detector = hist_detector / np.sum(hist_detector)
    energy = np.cumsum(hist_detector)
    index = np.argmin(np.abs(energy - 1 + percent))
    threshold = center[index]
    return threshold


def likelihood_ratio_single_Pd(Para2, thres, seg_Pv):
    tmask = Para2 > thres
    mask1 = tmask & (~seg_Pv)
    mu1 = aver_single_one(Para2, mask1)
    n1 = np.sum(mask1)
    mask2 = (~tmask) & (~seg_Pv)
    mu2 = aver_single_one(Para2, mask2)
    n2 = np.sum(mask2)
    mask0 = ~seg_Pv
    mu = aver_single_one(Para2, mask0)
    n = np.sum(mask0)
    pratio = -n1 * np.log(mu1) - n2 * np.log(mu2) + n * np.log(mu)
    return pratio


def newthres(input_data, filename):
    if 'terrasar' in filename:
        max_factor1 = 30
        max_factor2 = 30
    else:
        max_factor1 = 20
        max_factor2 = 20

    xsize = input_data.shape[0]
    ysize = input_data.shape[1]
    temp = np.ones((win, win)) / win / win
    output_data = np.zeros_like(input_data)
    for channel in range(3):
        output_data[:, :, channel, channel] = convolve2d(input_data[:, :, channel, channel], temp, mode='same',
                                                         boundary='symm')
    Ps, Pd, Pv = freeman_decom_T(output_data)
    data = input_data[:, :, 0, 0] + input_data[:, :, 1, 1] + input_data[:, :, 2, 2]
    Para1 = Pv
    Para2 = Pd
    mask_Pd_neg = Para2 < 0
    Para2[mask_Pd_neg] = 0
    mask_Pd_pos = Para2 > np.max(data)
    Para2[mask_Pd_pos] = np.max(data)
    start_time = time.time()
    rwin = (nWindow - 1) // 2
    image_avg1, image_var1 = var_parallel2(Para1, nWindow)
    image_select1 = image_avg1[rwin:-rwin, rwin:-rwin] * image_var1[rwin:-rwin, rwin:-rwin]
    val1, idx1 = np.min(image_select1), np.argmin(image_select1)
    i1, j1 = np.unravel_index(idx1, image_select1.shape)
    t2 = time.time() - start_time
    Sample_box1 = [j1, i1, nWindow, nWindow]
    smask1 = np.zeros((xsize, ysize), dtype=bool)
    smask1[i1:i1 + nWindow, j1:j1 + nWindow] = 1
    smean = np.mean(Para1[smask1])
    if thres_type == 4:
        thres_pv = smean
        pratio = likelihood_ratio_single(Para1, thres_pv)
        for factor1 in range(min_factor1, max_factor1 + 1):
            tthres_pv = 10 ** (factor1 / 10) * smean
            pratio_temp = likelihood_ratio_single(Para1, tthres_pv)
            if pratio_temp > pratio:
                pratio = pratio_temp
                thres_pv = tthres_pv
    elif thres_type == 3:
        thres_pv = gamma_thres(Para1[smask1], pfa)
        pdf_fitting_single(Para1, smask1)
    elif thres_type == 2:
        thres_pv = thres_deter(Para1, smask1, pfa)
    elif thres_type == 1:
        thres_pv = factor0 * smean
    else:
        thres_pv = 0.02
    seg_Pv = Para1 < thres_pv
    image_avg2, image_var2 = var_parallel2(Para2, nWindow)
    image_select2 = image_avg2[rwin + 1:-rwin, rwin + 1:-rwin]
    val2, idx2 = np.max(image_select2), np.argmax(image_select2)
    i2, j2 = np.unravel_index(idx2, image_select2.shape)
    Sample_box2 = [j2, i2, nWindow, nWindow]
    smask2 = np.zeros((xsize, ysize), dtype=bool)
    smask2[i2:i2 + nWindow, j2:j2 + nWindow] = 1
    smean2 = np.mean(Para2[smask2])
    if thres_type == 4:
        thres_pd = smean2 / (10 ** (min_factor2 / 10))
        pratio2 = likelihood_ratio_single_Pd(Para2, thres_pd, seg_Pv)
        for factor2 in range(min_factor2 + 1, max_factor2 + 1):
            tthres_pd = smean2 / (10 ** (factor2 / 10))
            pratio_temp2 = likelihood_ratio_single_Pd(Para2, tthres_pd, seg_Pv)
            if pratio_temp2 < pratio2:
                break
            else:
                pratio2 = pratio_temp2
                thres_pd = tthres_pd
    elif thres_type == 3:
        thres_pd = gamma_thres(Para2[smask2], pfa)
        pdf_fitting_single(Para2, smask2)
    elif thres_type == 2:
        thres_pd = thres_deter(Para2, smask2, pfa)
    elif thres_type == 1:
        thres_pd = factor0 * smean2
    else:
        thres_pd = 0.02
    seg_Pd = Para2 > thres_pd
    seg_Pd = seg_Pd & (~seg_Pv)

    X0 = 2 * np.ones((xsize, ysize))
    X0[seg_Pv] = 1
    X0[seg_Pd] = 3
    return seg_Pv,seg_Pd,X0, data, Para1, Para2, thres_pv, thres_pd


def newthresSAR(input_data, filename):
    if 'terrasar' in filename:
        max_factor1 = 30
        max_factor2 = 30
    else:
        max_factor1 = 20
        max_factor2 = 20
    xsize = input_data.shape[0]
    ysize = input_data.shape[1]
    Pd= input_data
    Pv = input_data
    data = input_data
    Para1 = Pv
    Para2 = Pd
    mask_Pd_neg = Para2 < 0
    Para2[mask_Pd_neg] = 0
    mask_Pd_pos = Para2 > np.max(data)
    Para2[mask_Pd_pos] = np.max(data)
    start_time = time.time()
    rwin = (nWindow - 1) // 2
    image_avg1, image_var1 = var_parallel2(Para1, nWindow)
    image_select1 = image_avg1[rwin:-rwin, rwin:-rwin] * image_var1[rwin:-rwin, rwin:-rwin]
    val1, idx1 = np.min(image_select1), np.argmin(image_select1)
    i1, j1 = np.unravel_index(idx1, image_select1.shape)
    t2 = time.time() - start_time
    Sample_box1 = [j1, i1, nWindow, nWindow]
    smask1 = np.zeros((xsize, ysize), dtype=bool)
    smask1[i1:i1 + nWindow, j1:j1 + nWindow] = 1
    smean = np.mean(Para1[smask1])
    if thres_type == 4:
        thres_pv = smean
        pratio = likelihood_ratio_single(Para1, thres_pv)
        for factor1 in range(min_factor1, max_factor1 + 1):
            tthres_pv = 10 ** (factor1 / 10) * smean
            pratio_temp = likelihood_ratio_single(Para1, tthres_pv)
            if pratio_temp > pratio:
                pratio = pratio_temp
                thres_pv = tthres_pv
    elif thres_type == 3:
        thres_pv = gamma_thres(Para1[smask1], pfa)
        pdf_fitting_single(Para1, smask1)
    elif thres_type == 2:
        thres_pv = thres_deter(Para1, smask1, pfa)
    elif thres_type == 1:
        thres_pv = factor0 * smean
    else:
        thres_pv = 0.02
    seg_Pv = Para1 < thres_pv
    image_avg2, image_var2 = var_parallel2(Para2, nWindow)
    image_select2 = image_avg2[rwin + 1:-rwin, rwin + 1:-rwin]
    val2, idx2 = np.max(image_select2), np.argmax(image_select2)
    i2, j2 = np.unravel_index(idx2, image_select2.shape)
    Sample_box2 = [j2, i2, nWindow, nWindow]
    smask2 = np.zeros((xsize, ysize), dtype=bool)
    smask2[i2:i2 + nWindow, j2:j2 + nWindow] = 1
    smean2 = np.mean(Para2[smask2])
    if thres_type == 4:
        thres_pd = smean2 / (10 ** (min_factor2 / 10))
        pratio2 = likelihood_ratio_single_Pd(Para2, thres_pd, seg_Pv)
        for factor2 in range(min_factor2 + 1, max_factor2 + 1):
            tthres_pd = smean2 / (10 ** (factor2 / 10))
            pratio_temp2 = likelihood_ratio_single_Pd(Para2, tthres_pd, seg_Pv)
            if pratio_temp2 < pratio2:
                break
            else:
                pratio2 = pratio_temp2
                thres_pd = tthres_pd
    elif thres_type == 3:
        thres_pd = gamma_thres(Para2[smask2], pfa)
        pdf_fitting_single(Para2, smask2)
    elif thres_type == 2:
        thres_pd = thres_deter(Para2, smask2, pfa)
    elif thres_type == 1:
        thres_pd = factor0 * smean2
    else:
        thres_pd = 0.02
    seg_Pd = Para2 > thres_pd
    seg_Pd = seg_Pd & (~seg_Pv)

    X0 = 2 * np.ones((xsize, ysize))
    X0[seg_Pv] = 1
    X0[seg_Pd] = 3
    return seg_Pv,seg_Pd,X0, data, Para1, Para2, thres_pv, thres_pd