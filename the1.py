import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

hash_3d = {}
hash_pc = {}

support = []
support_names = []
def calculate_perchannel_histogram(im, nbins, name):
    if hash_pc.get(name) != None:
        return hash_pc.get(name)
    q = int(256/nbins)

    blue = np.zeros(nbins)
    green = np.zeros(nbins)
    red = np.zeros(nbins)
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
    if hash_3d.get(name) != None:
        return hash_3d.get(name)
    q = int(256/nbins)
    h = np.zeros(pow(nbins, 3))
    #im = im.ravel()
    h = h.ravel()
    im = im // q

    for i in range(nbins ** 3):
        h[i] += (im[:, :, 0] == i).sum()
        h[i] += (im[:, :, 1] == i).sum()
        h[i] += (im[:, :, 2] == i).sum()

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
    res = h/n

    res[np.where(res == 0)] = 0.00001

    return res

def check_names(name1, name2):
    if name1.split('/')[2] == name2.split('/')[2]:
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

        grid_result_q = []
        grid_list_q = grid(gridSize, qname)
        if thrreD:
            if gridSize == 1:
                qHist = calculate_3d_histogram(qname, nbins, query_names[count_q])
            else:
                for i in range(gridSize*gridSize):
                    qHist = calculate_3d_histogram(grid_list_q[i], nbins, query_names[count_q]  + str(gridSize) + str(i))  # File name + i in order to indicate the grid
                    grid_result_q.append(qHist)
        else:
            if gridSize == 1:
                qHist = calculate_perchannel_histogram(qname, nbins, query_names[count_q])
            else:
                for i in range(gridSize*gridSize):
                    qHist = calculate_perchannel_histogram(grid_list_q[i], nbins, query_names[count_q] + str(gridSize) + str(i))
                    grid_result_q.append(qHist)

        for sname in support:
            grid_list_s = grid(gridSize, sname)
            cumulative_jsd = 0

            if thrreD:
                if gridSize == 1:
                    sHist = calculate_3d_histogram(sname, nbins, support_names[count_s])
                    jsd = calculate_jsd(sHist, qHist, nbins)

                else:
                    for i in range(gridSize * gridSize):
                        sHist = calculate_3d_histogram(sname, nbins, support_names[count_s] + str(gridSize) + str(i))
                        jsdB = calculate_jsd(sHist[0], grid_result_q[i][0], nbins)  # B
                        jsdG = calculate_jsd(sHist[1], grid_result_q[i][1], nbins)  # G
                        jsdR = calculate_jsd(sHist[2], grid_result_q[i][2], nbins)  # R
                        jsd = (jsdB + jsdG + jsdR) / 3

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
                    for i in range(gridSize*gridSize):
                        sHist = calculate_perchannel_histogram(grid_list_s[i], nbins, support_names[count_s] + str(gridSize) + str(i))

                        jsdB = calculate_jsd(sHist[0], grid_result_q[i][0], nbins)  # B
                        jsdG = calculate_jsd(sHist[1], grid_result_q[i][1], nbins)  # G
                        jsdR = calculate_jsd(sHist[2], grid_result_q[i][2], nbins)  # R
                        jsd = (jsdB + jsdG + jsdR) / 3

                        cumulative_jsd += jsd
                    jsd = cumulative_jsd / (gridSize*gridSize)      # Take average

                if jsd < minJSD:
                    minJSD = jsd
                    minS = support_names[count_s]
            count_s += 1

        if check_names(minS, query_names[count_q]):
            correct_retrieve += 1
        count_q += 1
    top1_acc = correct_retrieve / total
    return top1_acc


def grid(n, im):
    height = im.shape[0]
    grids = []
    for i in range(n):
        for j in range(n):
            x = im[i*n: (i+1)*n, j*n:(j+1)*n]
            grids.append(x)
    return grids

def main():
#    im = cv.imread("dataset/support_96/Acadian_Flycatcher_0016_887710060.jpg")
    file = open("dataset/InstanceNames.txt", "r" )
    nameList = file.readlines()
    nameList = [x[:-1] for x in nameList[:-1]]   # delete \n at the end but not delete from the last one !
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
    out = open("result.txt", "w")

    # # Question 1
    #     #Query 1
    # question_1_16 = query_compare(support, q_1, q_1_names, 16, nameList, True, 1)
    # question_1_8 = query_compare(support, q_1, q_1_names, 8, nameList, True, 1)
    # question_1_4 = query_compare(support, q_1, q_1_names, 4, nameList, True, 1)
    # question_1_2 = query_compare(support, q_1, q_1_names, 2, nameList, True, 1)
    # print("1_2" , question_1_2)
    #     #Query 2
    # question_1_16 = query_compare(support, q_2, q_2_names, 16, nameList, True, 1)
    # question_1_8 = query_compare(support, q_2, q_2_names, 8, nameList, True, 1)
    # question_1_4 = query_compare(support, q_2, q_2_names, 4, nameList, True, 1)
    # question_1_2 = query_compare(support, q_2, q_2_names, 2, nameList, True, 1)
    #     #Query 3
    # question_1_16 = query_compare(support, q_3, q_3_names, 16, nameList, True, 1)
    # question_1_8 = query_compare(support, q_3, q_3_names, 8, nameList, True, 1)
    # question_1_4 = query_compare(support, q_3, q_3_names, 4, nameList, True, 1)
    # question_1_2 = query_compare(support, q_3, q_3_names, 2, nameList, True, 1)
    #
    # # Question 2
    #     #Query 1
    # question_2_16 = query_compare(support, q_1, q_1_names, 16, nameList, False, 1)
    # question_2_8 = query_compare(support, q_1, q_1_names, 8, nameList, False, 1)
    # question_2_4 = query_compare(support, q_1, q_1_names, 4, nameList, False, 1)
    # question_2_2 = query_compare(support, q_1, q_1_names, 2, nameList, False, 1)
    # question_2_1 = query_compare(support, q_1, q_1_names, 8, nameList, False, 1)
    # print("2_1", question_2_1)
    #     #Query 2
    # question_2_16 = query_compare(support, q_2, q_2_names, 16, nameList, False, 1)
    # question_2_8 = query_compare(support, q_2, q_2_names, 8, nameList, False, 1)
    # question_2_4 = query_compare(support, q_2, q_2_names, 4, nameList, False, 1)
    # question_2_2 = query_compare(support, q_2, q_2_names, 2, nameList, False, 1)
    # question_2_1 = query_compare(support, q_2, q_2_names, 1, nameList, False, 1)
    #     #Query 3
    # question_2_16 = query_compare(support, q_3, q_3_names, 16, nameList, False, 1)
    # question_2_8 = query_compare(support, q_3, q_3_names, 8, nameList, False, 1)
    # question_2_4 = query_compare(support, q_3, q_3_names, 4, nameList, False, 1)
    # question_2_2 = query_compare(support, q_3, q_3_names, 2, nameList, False, 1)
    # question_2_1 = query_compare(support, q_3, q_3_names, 1, nameList, False, 1)
    #
    # # Question 3
    #     #2x2
    # question_3_1_3D = query_compare(support, q_1, q_1_names, 16, nameList, True, 2)
    question_3_1_pc = query_compare(support, q_1, q_1_names, 16, nameList, False, 2)
    print("Query 1: " ,question_3_1_pc)
    #     #4x4
    # question_3_2_3D = query_compare(support, q_1, q_1_names, 16, nameList, True, 4)
    # question_3_2_pc = query_compare(support, q_1, q_1_names, 16, nameList, False, 4)
    #     #6x6
    # question_3_3_3D = query_compare(support, q_1, q_1_names, 16, nameList, True, 8)
    # question_3_3_pc = query_compare(support, q_1, q_1_names, 16, nameList, False, 8)
    #     #8x8
    # question_3_4_3D = query_compare(support, q_1, q_1_names, 16, nameList, True, 16)
    # question_3_4_pc = query_compare(support, q_1, q_1_names, 16, nameList, False, 16)
    #
    # # Question 4
    #     #2x2
    # question_4_1_3D = query_compare(support, q_2, q_2_names, 16, nameList, True, 2)
    # question_4_1_pc = query_compare(support, q_2, q_2_names, 16, nameList, False, 2)
    #     #4x4
    # question_4_2_3D = query_compare(support, q_2, q_2_names, 16, nameList, True, 4)
    # question_4_2_pc = query_compare(support, q_2, q_2_names, 16, nameList, False, 4)
    #     #6x6
    # question_4_3_3D = query_compare(support, q_2, q_2_names, 16, nameList, True, 8)
    # question_4_3_pc = query_compare(support, q_2, q_2_names, 16, nameList, False, 8)
    #     #8x8
    # question_4_4_3D = query_compare(support, q_2, q_2_names, 16, nameList, True, 16)
    # question_4_4_pc = query_compare(support, q_2, q_2_names, 16, nameList, False, 16)
    #
    # # Question 5
    #     #2x2
    # question_5_1_3D = query_compare(support, q_3, q_3_names, 16, nameList, True, 2)
    # question_5_1_pc = query_compare(support, q_3, q_3_names, 16, nameList, False, 2)
    #     #4x4
    # question_5_2_3D = query_compare(support, q_3, q_3_names, 16, nameList, True, 4)
    # question_5_2_pc = query_compare(support, q_3, q_3_names, 16, nameList, False, 4)
    #     #6x6
    # question_5_3_3D = query_compare(support, q_3, q_3_names, 16, nameList, True, 8)
    # question_5_3_pc = query_compare(support, q_3, q_3_names, 16, nameList, False, 8)
    #     #8x8
    # question_5_4_3D = query_compare(support, q_3, q_3_names, 16, nameList, True, 16)
    # question_5_4_pc = query_compare(support, q_3, q_3_names, 16, nameList, False, 16)


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