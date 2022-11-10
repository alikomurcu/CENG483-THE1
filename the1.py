import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

hash_3d = {}
hash_pc = {}

support = []
support_names = []
def calculate_perchannel_histogram(im, nbins, name):
    if name in hash_pc.keys():
        return hash_pc.get(name)
    q = int(256/nbins)

    blue = np.zeros(nbins)
    green = np.zeros(nbins)
    red = np.zeros(nbins)
    # im = im/q
    # blue = (im[:,:,0]/q == np.arange(nbins)).sum()
    # green = (im[:,:,1]/q == np.arange(nbins)).sum()
    # red = (im[:,:,2]/q == np.arange(nbins)).sum()
    for i in range(nbins):
        blue[i] = (im[:,:,0] // q == i).sum()
        green[i] = (im[:,:,1] // q == i).sum()
        red[i] = (im[:,:,2] // q == i).sum()
    B = l1_normalization(blue)
    G = l1_normalization(green)
    R = l1_normalization(red)
    hash_pc[name] = [B, G, R]
    return [B,G,R]

def calculate_3d_histogram(im, nbins, name):
    if name in hash_3d.keys():
        return hash_3d.get(name)
    q = 256 // nbins
    h = np.zeros(nbins * nbins * nbins, int)
    im = np.reshape(im, (im.shape[0] * im.shape[0], 3))
    im = np.floor_divide(im, q)
    # base arithmethic with base-3 applied
    first = im[:,0]                         # base __*
    second = im[:,1] * nbins                # base _*_
    third = im[:,2] * (nbins * nbins)       # base *__

    x = first + second + third

    uniqList, count = np.unique(x, return_counts = True)

    h[uniqList] = count

    l_h = l1_normalization(h)
    hash_3d[name] = l_h
    return l_h    # TODO: add l1_normalization here


def calculate_jsd(s, q, n):
    return 0.5 * (calculate_kl(q, (s+q)/2) + calculate_kl(s, (s+q)/2))

def calculate_kl(s, q):
    return q.dot(np.log(q/s))

def l1_normalization(h):    # For 1D
    n = np.linalg.norm(h, 1)

    if n == 0:
        n = 0.00001
    normalized = h/n
    normalized[normalized == 0] = 0.00001
    return normalized

def check_names(name1, name2, grid):
    n1 = name1.split('/')[2].split('.')[0]
    n2 = name2.split('/')[2].split('.')[0]
    if n1 == n2:
        return True
    return False

def query_compare(support, query, query_names, nbins, nameList, thrreD, gridSize):
    correct_retrieve = 0
    total = len(nameList)
    count_q = 0
    minS = ""
    for qname in query:
        minJSD = 99999999999999999 # some large number TODO: look here! maybe flt_max
        count_s = 0

        histograms_q = []
        grid_list_q = grid(gridSize, qname)
        if thrreD:
            if gridSize == 1:
                qHist = calculate_3d_histogram(qname, nbins, query_names[count_q])
            else:
                for i in range(gridSize*gridSize):
                    qHist = calculate_3d_histogram(grid_list_q[i], nbins, query_names[count_q] + str(gridSize) + "q" + str(i))  # File name + i in order to indicate the grid
                    histograms_q.append(qHist)
        else:
            if gridSize == 1:
                qHist = calculate_perchannel_histogram(qname, nbins, query_names[count_q])
            else:
                for i in range(gridSize*gridSize):
                    qHist = calculate_perchannel_histogram(grid_list_q[i], nbins, query_names[count_q] + str(gridSize) + "q" + str(i))
                    histograms_q.append(qHist)

        for sname in support:
            grid_list_s = grid(gridSize, sname)

            if thrreD:
                if gridSize == 1:
                    sHist = calculate_3d_histogram(sname, nbins, support_names[count_s])
                    jsd = calculate_jsd(sHist, qHist, nbins)

                else:
                    cumulative_jsd = 0
                    for i in range(gridSize * gridSize):
                        sHist = calculate_3d_histogram(sname, nbins, support_names[count_s] + str(gridSize) + "s" + str(i))
                        jsd = calculate_jsd(sHist, histograms_q[i], nbins)  # B

                        cumulative_jsd += jsd
                    jsd = cumulative_jsd / (gridSize*gridSize)      # Take average
                if jsd < minJSD:
                    minJSD = jsd
                    minS = support_names[count_s]
            else:
                if gridSize == 1:
                    sHist = calculate_perchannel_histogram(sname, nbins, support_names[count_s])
                    jsdB = calculate_jsd(sHist[0], qHist[0], nbins)  # B
                    jsdG = calculate_jsd(sHist[1], qHist[1], nbins)  # G
                    jsdR = calculate_jsd(sHist[2], qHist[2], nbins)  # R
                    jsd = (jsdB + jsdG + jsdR) / 3

                else:
                    cumulative_jsd = 0
                    for i in range(gridSize*gridSize):
                        sHist = calculate_perchannel_histogram(grid_list_s[i], nbins, support_names[count_s] + str(gridSize) + "s" + str(i))

                        jsdB = calculate_jsd(sHist[0], histograms_q[i][0], nbins)  # B
                        jsdG = calculate_jsd(sHist[1], histograms_q[i][1], nbins)  # G
                        jsdR = calculate_jsd(sHist[2], histograms_q[i][2], nbins)  # R
                        jsd = (jsdB + jsdG + jsdR) / 3

                        cumulative_jsd += jsd
                    jsd = cumulative_jsd / (gridSize*gridSize)      # Take average

                if jsd < minJSD:
                    minJSD = jsd
                    minS = support_names[count_s]
            count_s += 1

        if check_names(minS, query_names[count_q], grid != 1):
            correct_retrieve += 1
        count_q += 1
    top1_acc = correct_retrieve / total
    return top1_acc


def grid(n, im):
    sz = im.shape[0] // n
    grids = []
    for i in range(n):
        for j in range(n):
            x = im[i*sz: (i+1)*sz, j*sz:(j+1)*sz]
            grids.append(x)
    return grids

def main():
#    im = cv.imread("dataset/support_96/Acadian_Flycatcher_0016_887710060.jpg")
    file = open("dataset/InstanceNames.txt", "r" )
    nameList = file.readlines()
    nameList = [x.strip() for x in nameList]   # delete \n at the end but not delete from the last one !
    file.close()


    # Support
    for name in nameList:
        name = "dataset/support_96/" + name
        im = cv.imread(name)
        support.append(im)
        support_names.append(name)
    q_1 = []
    q_1_names = []
    # Query 1
    for name in nameList:
        name = "dataset/query_1/" + name
        im = cv.imread(name)
        q_1.append(im)
        q_1_names.append(name)
    q_2 = []
    q_2_names = []
    # Query 2
    for name in nameList:
        name = "dataset/query_2/" + name
        im = cv.imread(name)
        q_2.append(im)
        q_2_names.append(name)
    q_3 = []
    q_3_names = []
    # Query 3
    for name in nameList:
        name = "dataset/query_3/" + name
        f = open(name, "r")
        im = cv.imread(name)
        q_3.append(im)
        q_3_names.append(name)

    q = [q_1, q_2, q_3]
    qnames = [q_1_names, q_2_names, q_3_names]
    out = open("result.txt", "w")
    outStr = ""
    # # Question 1
    outStr += "Question1 - 3D histograms \n"
    #     #Query 1
    question_1_16 = query_compare(support, q_1, q_1_names, 16, nameList, True, 1)
    outStr += "Top-1 Acc of Query 1 16 bins " + str(question_1_16) + " \n"
    question_1_8 = query_compare(support, q_1, q_1_names, 8, nameList, True, 1)
    outStr += "Top-1 Acc of Query 1 8 bins " + str(question_1_8) + " \n"
    question_1_4 = query_compare(support, q_1, q_1_names, 4, nameList, True, 1)
    outStr += "Top-1 Acc of Query 1 4 bins " + str(question_1_4) + " \n"
    question_1_2 = query_compare(support, q_1, q_1_names, 2, nameList, True, 1)
    outStr += "Top-1 Acc of Query 1 2 bins " + str(question_1_2) + " \n"
    # print("1_2" , question_1_2)
    #     #Query 2
    question_1_16 = query_compare(support, q_2, q_2_names, 16, nameList, True, 1)
    outStr += "Top-1 Acc of Query 2 with 16 bins " + str(question_1_16) + " \n"
    question_1_8 = query_compare(support, q_2, q_2_names, 8, nameList, True, 1)
    outStr += "Top-1 Acc of Query 2 with 8 bins " + str(question_1_8) + " \n"
    question_1_4 = query_compare(support, q_2, q_2_names, 4, nameList, True, 1)
    outStr += "Top-1 Acc of Query 2 with 4 bins " + str(question_1_4) + " \n"
    question_1_2 = query_compare(support, q_2, q_2_names, 2, nameList, True, 1)
    outStr += "Top-1 Acc of Query 2 with 2 bins " + str(question_1_2) + " \n"
        #Query 3
    question_1_16 = query_compare(support, q_3, q_3_names, 16, nameList, True, 1)
    outStr += "Top-1 Acc of Query 3 with 16 bins " + str(question_1_16) + " \n"
    question_1_8 = query_compare(support, q_3, q_3_names, 8, nameList, True, 1)
    outStr += "Top-1 Acc of Query 3 with 8 bins " + str(question_1_8) + " \n"
    question_1_4 = query_compare(support, q_3, q_3_names, 4, nameList, True, 1)
    outStr += "Top-1 Acc of Query 3 with 4 bins " + str(question_1_4) + " \n"
    question_1_2 = query_compare(support, q_3, q_3_names, 2, nameList, True, 1)
    outStr += "Top-1 Acc of Query 3 with 2 bins " + str(question_1_2) + " \n"

    # Question 2
    outStr += "Question2 - PC histograms \n"
    #     #Query 1
    for i in range(2):  # question 1 & question 2
        threeD = True if i == 0 else False
        for j in range(3):  # query 1 2 3
            if i == 1:  # if Question 2
                q32 = query_compare(support, q[j], qnames[j], 32, nameList, threeD, 1)
                outStr += "Top-1 Acc of Query 1 with 32 bins " + str(q32) + "\n"
            q16 = query_compare(support, q[j], qnames[j], 16, nameList, threeD, 1)
            outStr += "Top-1 Acc of Query 1 with 16 bins " + str(q16) + "\n"
            q8 = query_compare(support, q[j], qnames[j], 8, nameList, threeD, 1)
            outStr += "Top-1 Acc of Query 1 with 8 bins " + str(q8) + "\n"
            q4 = query_compare(support, q[j], qnames[j], 4, nameList, threeD, 1)
            outStr += "Top-1 Acc of Query 1 with 4 bins " + str(q4) + "\n"
            q2 = query_compare(support, q[j], qnames[j], 2, nameList, threeD, 1)
            outStr += "Top-1 Acc of Query 1 with 2 bins " + str(q2) + "\n"
    #Query 2

    # question_2_1 = query_compare(support, q_1, q_1_names, 32, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 1 with 32 bins " + str(question_2_1) + "\n"
    # question_2_16 = query_compare(support, q_1, q_1_names, 16, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 1 with 16 bins " + str(question_2_16) + "\n"
    # question_2_8 = query_compare(support, q_1, q_1_names, 8, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 1 with 8 bins " + str(question_2_8) + "\n"
    # question_2_4 = query_compare(support, q_1, q_1_names, 4, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 1 with 4 bins " + str(question_2_4) + "\n"
    # question_2_2 = query_compare(support, q_1, q_1_names, 2, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 1 with 2 bins " + str(question_2_2) + "\n"    #Query 2
    #
    # question_2_1 = query_compare(support, q_2, q_2_names, 32, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 2 with 32 bins " + str(question_2_1) + "\n"
    # question_2_16 = query_compare(support, q_2, q_2_names, 16, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 2 with 16 bins " + str(question_2_16) + "\n"
    # question_2_8 = query_compare(support, q_2, q_2_names, 8, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 2 with 8 bins " + str(question_2_8) + "\n"
    # question_2_4 = query_compare(support, q_2, q_2_names, 4, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 2 with 4 bins " + str(question_2_4) + "\n"
    # question_2_2 = query_compare(support, q_2, q_2_names, 2, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 2 with 2 bins " + str(question_2_2) + "\n"    #Query 3
    #
    # question_2_1 = query_compare(support, q_3, q_3_names, 32, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 3 with 32 bins " + str(question_2_1) + "\n"
    # question_2_16 = query_compare(support, q_3, q_3_names, 16, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 3 with 16 bins " + str(question_2_16) + "\n"
    # question_2_8 = query_compare(support, q_3, q_3_names, 8, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 3 with 8 bins " + str(question_2_8) + "\n"
    # question_2_4 = query_compare(support, q_3, q_3_names, 4, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 3 with 4 bins " + str(question_2_4) + "\n"
    # question_2_2 = query_compare(support, q_3, q_3_names, 2, nameList, False, 1)
    # outStr += "Top-1 Acc of Query 3 with 2 bins " + str(question_2_2) + "\n"
    # # Question 3
    #     #2x2
    numBins = 2
    for i in range(3):
        outStr += "Question " + str(i+3) + "\n"
        g2_3D = query_compare(support, q[i], qnames[i], numBins, nameList, True, 2)
        outStr += "Top-1 3D Acc of Query " + str(i+1) + " with x bins with grid 2 : " + str(g2_3D) + "\n"
        g2_PC = query_compare(support, q[i], qnames[i], numBins, nameList, True, 2)
        outStr += "Top-1 PC Acc of Query " + str(i+1) + " with x bins with grid 2 : " + str(g2_PC) + "\n"
            #4x4
        g4_3D = query_compare(support, q[i], qnames[i], numBins, nameList, True, 4)
        outStr += "Top-1 3D Acc of Query " + str(i+1) + " with x bins with grid 4 : " + str(g4_3D) + "\n"
        g4_PC = query_compare(support, q[i], qnames[i], numBins, nameList, False, 4)
        outStr += "Top-1 PC Acc of Query " + str(i+1) + " with x bins with grid 4 : " + str(g4_PC) + "\n"
            #6x6
        g6_3D = query_compare(support, q[i], qnames[i], numBins, nameList, True, 8)
        outStr += "Top-1 3D Acc of Query " + str(i+1) + " with x bins with grid 8 : " + str(g6_3D) + "\n"
        g6_PC = query_compare(support, q[i], qnames[i], numBins, nameList, False, 8)
        outStr += "Top-1 PC Acc of Query " + str(i+1) + " with x bins with grid 8 : " + str(g6_PC) + "\n"
            #8x8
        g8_3D = query_compare(support, q[i], qnames[i], numBins, nameList, True, 16)
        outStr += "Top-1 3D Acc of Query " + str(i+1) + " with x bins with grid 16 : " + str(g8_3D) + "\n"
        g8_PC = query_compare(support, q[i], qnames[i], numBins, nameList, False, 16)
        outStr += "Top-1 PC Acc of Query " + str(i+1) + " with x bins with grid 16 : " + str(g8_PC) + "\n"

    # Question 4
#         #2x2
#     question_4_1_3D = query_compare(support, q_2, q_2_names, 16, nameList, True, 2)
#     outStr += "Top-1 3D Acc of Query 2 with x bins with grid 2: " + str(question_4_1_3D) + "\n"
#     question_4_1_pc = query_compare(support, q_2, q_2_names, 16, nameList, False, 2)
#     outStr += "Top-1 PC Acc of Query 2 with x bins with grid 2: " + str(question_4_1_pc) + "\n"
#         #4x4
#     question_4_2_3D = query_compare(support, q_2, q_2_names, 16, nameList, True, 4)
#     outStr += "Top-1 3D Acc of Query 2 with x bins with grid 4: " + str(question_4_2_3D) + "\n"
#     question_4_2_pc = query_compare(support, q_2, q_2_names, 16, nameList, False, 4)
#     outStr += "Top-1 PC Acc of Query 2 with x bins with grid 4: " + str(question_4_2_pc) + "\n"
#         #6x6
#     question_4_3_3D = query_compare(support, q_2, q_2_names, 16, nameList, True, 8)
#     outStr += "Top-1 3D Acc of Query 2 with x bins with grid 8: " + str(question_4_3_3D) + "\n"
#     question_4_3_pc = query_compare(support, q_2, q_2_names, 16, nameList, False, 8)
#     outStr += "Top-1 PC Acc of Query 2 with x bins with grid 8 : " + str(question_4_3_pc) + "\n"
#         #8x8
#     question_4_4_3D = query_compare(support, q_2, q_2_names, 16, nameList, True, 16)
#     outStr += "Top-1 3D Acc of Query 2 with x bins with grid 16 : " + str(question_4_4_3D) + "\n"
#     question_4_4_pc = query_compare(support, q_2, q_2_names, 16, nameList, False, 16)
#     outStr += "Top-1 PC Acc of Query 2 with x bins with grid 16 : " + str(question_4_4_pc) + "\n"
#
#
# # Question 5
#         #2x2
#     question_5_1_3D = query_compare(support, q_3, q_3_names, 16, nameList, True, 2)
#     outStr += "Top-1 3D Acc of Query 2 with x bins with grid 2: " + str(question_5_1_3D) + "\n"
#     question_5_1_pc = query_compare(support, q_3, q_3_names, 16, nameList, False, 2)
#     outStr += "Top-1 PC Acc of Query 2 with x bins with grid 2: " + str(question_5_1_pc) + "\n"
#         #4x4
#     question_5_2_3D = query_compare(support, q_3, q_3_names, 16, nameList, True, 4)
#     outStr += "Top-1 3D Acc of Query 2 with x bins with grid 4: " + str(question_5_2_3D) + "\n"
#     question_5_2_pc = query_compare(support, q_3, q_3_names, 16, nameList, False, 4)
#     outStr += "Top-1 PC Acc of Query 2 with x bins with grid 4: " + str(question_5_2_pc) + "\n"
#         #6x6
#     question_5_3_3D = query_compare(support, q_3, q_3_names, 16, nameList, True, 8)
#     outStr += "Top-1 3D Acc of Query 2 with x bins with grid 8: " + str(question_5_3_3D) + "\n"
#     question_5_3_pc = query_compare(support, q_3, q_3_names, 16, nameList, False, 8)
#     outStr += "Top-1 PC Acc of Query 2 with x bins with grid 8 : " + str(question_5_3_pc) + "\n"
#         #8x8
#     question_5_4_3D = query_compare(support, q_3, q_3_names, 16, nameList, True, 16)
#     outStr += "Top-1 3D Acc of Query 2 with x bins with grid 16 : " + str(question_5_4_3D) + "\n"
#     question_5_4_pc = query_compare(support, q_3, q_3_names, 16, nameList, False, 16)
#     outStr += "Top-1 PC Acc of Query 2 with x bins with grid 16 : " + str(question_5_4_pc) + "\n"

    out.write(outStr)
# im = cv.imread("dataset/support_96/Green_tailed_Towhee_0015_136678400.jpg")
# a = calculate_3d_histogram(im, 2)
# # x = calculate_3d_histogram_2(im, 2)
# b = calculate_perchannel_histogram(im, 2)
# b = calculate_perchannel_histogram2(im, 2)
# # print("a")
start = time.time()
main()
end = time.time()
print("time: " , end - start)