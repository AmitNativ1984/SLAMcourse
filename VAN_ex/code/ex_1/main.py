import cv2
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt




DATA_PATH = r'./VAN_ex/data/dataset05/sequences/05/'
NUM_KEYPTS = 1000

def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + "image_0/" + img_name, 0)
    img2 = cv2.imread(DATA_PATH + "image_1/" + img_name, 0)
    return img1, img2


def detect_keyPts(img):
    orb = cv2.ORB_create(nfeatures=1000)
    keypts, descriptors = orb.detectAndCompute(img, None)
    return keypts, descriptors

def vertical_match_diff(kpts1, kpts2, matches):
    deviations = []

    for ind, match in enumerate(matches):
        ver_diff = np.abs(kpts1[match.queryIdx].pt[1] - kpts2[match.trainIdx].pt[1])
        deviations.append(ver_diff)

    return deviations

def get_matcherScore_inliers_outliers(matches, thres):
    matches_inliers = []
    matches_outliers =[]
    
    kpts1_inliers = []
    kpts1_outliers = []
    kpts2_inliers = []
    kpts2_outliers = []
    for ind, match in enumerate(matches):
        if match.distance <= thres:
            matches_inliers.append(match)
            kpts1_inliers.append(kpts1[match.queryIdx])
            kpts2_inliers.append(kpts2[match.trainIdx])
        else:
            matches_outliers.append(match)
            kpts1_outliers.append(kpts1[match.queryIdx])
            kpts2_outliers.append(kpts2[match.trainIdx])
        
    return matches_inliers, matches_outliers, kpts1_inliers, kpts1_outliers, kpts2_inliers, kpts2_outliers,


def get_rectified_inliers_outliers(kpts1, kpts2, matches, thres):
    match_inliers = []
    match_outliers = []
    kpts1_inliers = []
    kpts1_outliers = []
    kpts2_inliers = []
    kpts2_outliers = []
    for ind, match in enumerate(matches):
        ver_diff = np.abs(kpts1[match.queryIdx].pt[1] - kpts2[match.trainIdx].pt[1])
        if ver_diff <= thres:
            match_inliers.append(match)
            kpts1_inliers.append(kpts1[match.queryIdx])
            kpts2_inliers.append(kpts2[match.queryIdx])
        else:
            match_outliers.append(match)
            kpts1_outliers.append(kpts1[match.queryIdx])
            kpts2_outliers.append(kpts2[match.queryIdx])

    return match_inliers, match_outliers, kpts1_inliers, kpts1_outliers, kpts2_inliers, kpts2_outliers

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
    img1, img2 = read_images(idx)
    
    kpts1, desc1 = detect_keyPts(img1)
    kpts2, desc2 = detect_keyPts(img2)

    # draw key points
    img1bgr = np.dstack((img1, img1, img1))
    img2bgr = np.dstack((img2, img2, img2))
    cv2.drawKeypoints(img1bgr, kpts1, img1bgr, color=(255,0,0))
    cv2.drawKeypoints(img2bgr, kpts2, img2bgr, color=(0,0,255))
    cv2.imshow("keyPoints [left/right]", np.hstack((img1bgr, img2bgr)))
    

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
    cumsumThres = 0.88
    match_dist = []
    for m in matches:
        match_dist.append(m.distance)
        
    counts, bins = plot_histogram(match_dist)
    cumsum_counts = np.cumsum(counts)
    thres = np.argmax(cumsum_counts>cumsumThres)
    max_match_diff = bins[thres]

    matches_inliers, matches_outliers, kpts1_inliers, kpts1_outliers, kpts2_inliers, kpts2_outliers = get_matcherScore_inliers_outliers(matches, thres)
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

    # draw image
    cv2.imshow
    cv2.waitKey(0)