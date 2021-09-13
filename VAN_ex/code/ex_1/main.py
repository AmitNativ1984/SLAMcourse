import cv2
import os
print(os.getcwd())
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from VAN_ex.code.utils import *

DATA_PATH = r'./VAN_ex/data/dataset05/sequences/05/'
NUM_KEYPTS = 1000

def plot_histogram(x):
    counts, bins = np.histogram(x, bins=list(range(0, 100, 1)))
    counts = counts.astype(np.float)
    counts *= 1 / counts.sum()
    plt.figure(1)
    plt.bar(bins[:-1], counts)
    plt.ylim((0,1))

    return counts, bins

if __name__ == "__main__":
    idx = 0
    img1, img2 = read_images(idx, DATA_PATH)
    
    kpts1, desc1 = detect_keyPts(img1)
    kpts2, desc2 = detect_keyPts(img2)

    # draw key points
    draw_kpts(img1, img2, kpts1, kpts2, title="kpts[left/right]")    

    # printing description of first two images
    print(tabulate([[np.array(desc1[0]).reshape(-1, 1), np.array(desc2[0]).reshape(-1, 1)]], headers=['left', 'right']))

    # matching the two descriptors:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1,desc2)

    # sort matches in descending order
    print("\nmatching based on feature BFmatcher distance")
    print("=================================================================")
    matches = sorted(matches, key = lambda x:x.distance)

    # display first 20 points
    imgMatches = cv2.drawMatches(img1,kpts1,img2,kpts2,matches[:20],None)
    cv2.imshow("Matches", imgMatches)
    
    # since images are rectified, examining the vertical distance between matches
    deviations = vertical_match_diff(kpts1, kpts2, matches)
    counts, bins = plot_histogram(deviations)

    dev_percent = sum([counts[i] for i, bin in enumerate(bins[:-1]) if bin > 2])
    print("matched perctenage that deviate from more than 2 pixels: {:.2f}%".format(dev_percent * 100))

    # rejecting matches based on matcher distance (taking 88% of matches)
    cumsumThres = 0.4
    match_dist = []
    for m in matches:
        match_dist.append(m.distance)
        
    counts, bins = plot_histogram(match_dist)
    cumsum_counts = np.cumsum(counts)
    thres = np.argmax(cumsum_counts>cumsumThres)
    max_match_diff = bins[thres]

    matches_inliers, matches_outliers, kpts1_inliers, kpts1_outliers, kpts2_inliers, kpts2_outliers = get_matcherScore_inliers_outliers(matches, thres, kpts1, kpts2)
    print("matcher threshold = {}".format(max_match_diff))
    print("num of inliers: {}, num of outliers: {}".format(len(kpts1_inliers), len(kpts1_outliers)))

    # creating vertical distance distance for good matches
    print("\ncalculating using vertical 2 px distance if rectified image pair")
    print("=================================================================")
    deviations = vertical_match_diff(kpts1, kpts2, matches)
    counts, bins = plot_histogram(deviations)

    # rejecting outliers that exceed 2 pixels difference:
    match_inliers_rect, match_outliers_rect, kpts1_inliers_rect, kpts1_outliers_rect, kpts2_inliers_rect, kpts2_outliers_rect = get_rectified_inliers_outliers(kpts1, kpts2, matches, thres=2)

    img1bgr = np.dstack((img1, img1, img1))
    img2bgr = np.dstack((img2, img2, img2))
    cv2.drawKeypoints(img1bgr, kpts1_inliers_rect, img1bgr, color=(0,125,255))
    cv2.drawKeypoints(img2bgr, kpts2_inliers_rect, img2bgr, color=(0,125,255))

    cv2.drawKeypoints(img1bgr, kpts1_outliers_rect, img1bgr, color=(255,125,0))
    cv2.drawKeypoints(img2bgr, kpts2_outliers_rect, img2bgr, color=(255,125,0))

    cv2.imshow("Inliers/Outliers", np.hstack((img1bgr, img2bgr)))

    print("num of inliers: {}, num of outliers: {}".format(len(kpts1_inliers_rect), len(kpts1_outliers_rect)))
    print("\n")

    # calucating number of matches that maintains vertical distance,
    # but matcher distance is too large:
    count = 0
    for match in match_inliers_rect:
        if match.distance > max_match_diff:
            count += 1
        
    
    print("number of inliers with matcher score > {}: {}".format(max_match_diff, count))
    print("P(good vertical distance | bad matcher score) = {:.2f}%".format (count / len(match_inliers_rect) * 100))

    
    #(1.7)
    #------
    # read matrices
    K, M1, M2 = read_cameras(DATA_PATH)

    # solving linear list squares with SVD to calculate detected points in world coordiantes
    P = K @ M1
    Q = K @ M2
    
    # traingulating points:
    X = []
    
    for match in match_inliers_rect:
        p = np.array([kpts1[match.queryIdx].pt[0],
                      kpts1[match.queryIdx].pt[1],
                      1])
        
        q = np.array([kpts2[match.trainIdx].pt[0],
                      kpts2[match.trainIdx].pt[1],
                      1])

        x = traingulate_point(P, Q, p, q)
        X.append(x)

    X = np.array(X)

    
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, facecolors="None", color='blue')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')


    # comparing with opencv function:
    Xcv = []
    for match in match_inliers_rect:
        p = np.array([kpts1[match.queryIdx].pt[0],
                      kpts1[match.queryIdx].pt[1],
                      1])
        
        q = np.array([kpts2[match.trainIdx].pt[0],
                      kpts2[match.trainIdx].pt[1],
                      1])

        x = cv2.triangulatePoints(P, Q, kpts1[match.queryIdx].pt, kpts2[match.trainIdx].pt)
        Xcv.append(x/ x[-1])

    Xcv = np.array(Xcv).squeeze(-1)
    ax.scatter(Xcv[:, 0], Xcv[:, 1], Xcv[:, 2], marker='^', color='red', alpha=0.5, facecolors="None")
        

    # draw image
    cv2.imshow
    plt.show()
    cv2.waitKey(0)