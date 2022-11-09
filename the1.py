import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def calculate_perchannel_histogram(im, nbins):
    q = int(256/nbins)
    h = np.zeros(nbins * 3)
    h = np.reshape(h, (3, nbins))
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            val = im[row, col]
            v = val//q      # int division
            h[0][v[0]] += 1     # B
            h[1][v[1]] += 1     # G
            h[2][v[2]] += 1     # R
    return l1_normalization(h)


def calculate_3d_histogram(im, nbins):
    q = int(256/nbins)
    h = np.zeros(pow(nbins, 3))
    h = np.reshape(h, (nbins, nbins, nbins))
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            val = im[row, col]
            v = val//q      # int division
            h[v[0], v[1], v[2]] += 1
    return h

def calculate_jsd(s, q):
    return 0

def calculate_kl(s, q):
    return

def l1_normalization(h):    # For 1D
    n = np.linalg.norm(h, 1)
    if n == 0:
        return 0
    res = h/n

    return res


def main():
#    im = cv.imread("dataset/support_96/Acadian_Flycatcher_0016_887710060.jpg")
    file = open("dataset/InstanceNames.txt", "r" )
    nameList = file.readlines()
    nameList = [x[:-1] for x in nameList[:-1]]   # delete \n at the end but not delete from the last one !
    file.close()

    support = []
    # Support
    for name in nameList:
        name = "dataset/support_96/" + name
        f = open(name, "r")
        im = cv.imread(name)
        support.append(im)

    q_1 = []
    # Query 1
    for name in nameList:
        name = "dataset/query_1/" + name
        f = open(name, "r")
        im = cv.imread(name)
        q_1.append(im)

    q_2 = []
    # Query 2
    for name in nameList:
        name = "dataset/query_2/" + name
        f = open(name, "r")
        im = cv.imread(name)
        q_2.append(im)

    q_3 = []
    # Query 3
    for name in nameList:
        name = "dataset/query_3/" + name
        f = open(name, "r")
        im = cv.imread(name)
        q_3.append(im)

    # Question 1
    min_qList = []
    for s in support:
        minJSD = 99999999 # some large number TODO: look here! maybe flt_max
        for q in q_1:
            sHist = calculate_perchannel_histogram(s, 16)   # 16 bins
            qHist = calculate_perchannel_histogram(q, 16)
            jsd = calculate_jsd(sHist, qHist)
            if jsd < minJSD:
                minJSD = jsd
                minQ = q
        min_qList.append(minQ)
# im = cv.imread("dataset/support_96/Green_tailed_Towhee_0015_136678400.jpg")
# a = calculate_3d_histogram(im,2)
# b = calculate_perchannel_histogram(im, 2)
# print("a")
#main()