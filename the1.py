import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
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
            B = l1_normalization(h[0][v[0]])
            G = l1_normalization(h[1][v[1]])
            R = l1_normalization(h[2][v[2]])
    return (B,G,R)


def calculate_3d_histogram(im, nbins):
    q = int(256/nbins)
    h = np.zeros(pow(nbins, 3))
    h = np.reshape(h, (nbins, nbins, nbins))
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            val = im[row, col]
            v = val//q      # int division
            h[v[0], v[1], v[2]] += 1
    return h    # TODO: add l1_normalization here

def calculate_jsd(s, q, n):
    return 0.5 * (calculate_kl(q, (s+q)/2) + calculate_kl(s, (s+q)/2))

def calculate_kl(s, q):
    return q.dot(np.log(q/s))

def l1_normalization(h):    # For 1D
    n = np.linalg.norm(h, 1)
    if n == 0:
        return 0
    res = h/n

    return res

def query_compare(support, query, nbins, nameList, thrreD = False):
    correct_retrieve = 0
    total = len(nameList)
    for qname in query:
        qIm = cv.imread(qname)
        minJSD = 99999999 # some large number TODO: look here! maybe flt_max
        for sname in support:
            sIm = cv.imread(sname)
            if thrreD:
                sHist = calculate_3d_histogram(sIm, nbins)
                qHist = calculate_3d_histogram(qIm, nbins)

                jsd = calculate_jsd(sHist, qHist, nbins)
                if jsd < minJSD:
                    minJSD = jsd
                    minS = sname
            else:
                sHist = calculate_perchannel_histogram(sIm, nbins)
                qHist = calculate_perchannel_histogram(qIm, nbins)

                jsdB = calculate_jsd(sHist[0], qHist[0], nbins)  # B
                jsdG = calculate_jsd(sHist[1], qHist[1], nbins)  # G
                jsdR = calculate_jsd(sHist[2], qHist[2], nbins)  # R
                jsd = (jsdB + jsdG + jsdR) / 3

                if jsd < minJSD:
                    minJSD = jsd
                    minS = sname

        if minS == qname:
            correct_retrieve += 1
    top1_acc = correct_retrieve / total
    return top1_acc

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
        im = cv.imread(name)
        support.append(im)

    q_1 = []
    # Query 1
    for name in nameList:
        name = "dataset/query_1/" + name
        im = cv.imread(name)
        q_1.append(im)

    q_2 = []
    # Query 2
    for name in nameList:
        name = "dataset/query_2/" + name
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
        #Query 1
    question_1_16 = query_compare(support, q_1, 16, nameList, True)
    question_1_8 = query_compare(support, q_1, 8, nameList, True)
    question_1_4 = query_compare(support, q_1, 4, nameList, True)
    question_1_2 = query_compare(support, q_1, 2, nameList, True)
        #Query 2
    question_1_16 = query_compare(support, q_2, 16, nameList, True)
    question_1_8 = query_compare(support, q_2, 8, nameList, True)
    question_1_4 = query_compare(support, q_2, 4, nameList, True)
    question_1_2 = query_compare(support, q_2, 2, nameList, True)
        #Query 3
    question_1_16 = query_compare(support, q_3, 16, nameList, True)
    question_1_8 = query_compare(support, q_3, 8, nameList, True)
    question_1_4 = query_compare(support, q_3, 4, nameList, True)
    question_1_2 = query_compare(support, q_3, 2, nameList, True)

    # Question 2
        #Query 1
    question_2_16 = query_compare(support, q_1, 16, nameList, False)
    question_2_8 = query_compare(support, q_1, 8, nameList, False)
    question_2_4 = query_compare(support, q_1, 4, nameList, False)
    question_2_2 = query_compare(support, q_1, 2, nameList, False)
    question_2_1 = query_compare(support, q_1, 2, nameList, False)
        #Query 2
    question_2_16 = query_compare(support, q_2, 16, nameList, False)
    question_2_8 = query_compare(support, q_2, 8, nameList, False)
    question_2_4 = query_compare(support, q_2, 4, nameList, False)
    question_2_2 = query_compare(support, q_2, 2, nameList, False)
    question_2_1 = query_compare(support, q_2, 2, nameList, False)
        #Query 3
    question_2_16 = query_compare(support, q_3, 16, nameList, False)
    question_2_8 = query_compare(support, q_3, 8, nameList, False)
    question_2_4 = query_compare(support, q_3, 4, nameList, False)
    question_2_2 = query_compare(support, q_3, 2, nameList, False)
    question_2_1 = query_compare(support, q_3, 2, nameList, False)

    # Question 3
        #2x2
    question_3_1_3D = query_compare(support, q_1, 16, nameList, True)
    question_3_1_pc = query_compare(support, q_1, 16, nameList, False)
        #4x4
    question_3_2_3D = query_compare(support, q_1, 16, nameList, True)
    question_3_2_pc = query_compare(support, q_1, 16, nameList, False)
        #6x6
    question_3_3_3D = query_compare(support, q_1, 16, nameList, True)
    question_3_3_pc = query_compare(support, q_1, 16, nameList, False)
        #8x8
    question_3_4_3D = query_compare(support, q_1, 16, nameList, True)
    question_3_4_pc = query_compare(support, q_1, 16, nameList, False)

    # Question 4
        #2x2
    question_4_1_3D = query_compare(support, q_2, 16, nameList, True)
    question_4_1_pc = query_compare(support, q_2, 16, nameList, False)
        #4x4
    question_4_2_3D = query_compare(support, q_2, 16, nameList, True)
    question_4_2_pc = query_compare(support, q_2, 16, nameList, False)
        #6x6
    question_4_3_3D = query_compare(support, q_2, 16, nameList, True)
    question_4_3_pc = query_compare(support, q_2, 16, nameList, False)
        #8x8
    question_4_4_3D = query_compare(support, q_2, 16, nameList, True)
    question_4_4_pc = query_compare(support, q_2, 16, nameList, False)

    # Question 5
        #2x2
    question_5_1_3D = query_compare(support, q_3, 16, nameList, True)
    question_5_1_pc = query_compare(support, q_3, 16, nameList, False)
        #4x4
    question_5_2_3D = query_compare(support, q_3, 16, nameList, True)
    question_5_2_pc = query_compare(support, q_3, 16, nameList, False)
        #6x6
    question_5_3_3D = query_compare(support, q_3, 16, nameList, True)
    question_5_3_pc = query_compare(support, q_3, 16, nameList, False)
        #8x8
    question_5_4_3D = query_compare(support, q_3, 16, nameList, True)
    question_5_4_pc = query_compare(support, q_3, 16, nameList, False)


im = cv.imread("dataset/support_96/Green_tailed_Towhee_0015_136678400.jpg")
a = calculate_3d_histogram(im,2)
b = calculate_perchannel_histogram(im, 2)
# # print("a")
# start = time.time()
# main()
# end = time.time()
# print(end - start)