from operator import truediv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def read_images(idx, DATA_PATH):
    img_name = '{:06d}.png'.format(idx)
    img1str = DATA_PATH + "image_0/" + img_name
    img1 = cv2.imread(img1str, 0)
    
    img2str = DATA_PATH + "image_1/" + img_name
    img2 = cv2.imread(img2str, 0)
    return img1, img2, img1str, img2str

def draw_kpts(img1, img2, kpts1, kpts2, title="kpts[left/right]"):
    img1bgr = np.dstack((img1, img1, img1))
    img2bgr = np.dstack((img2, img2, img2))
    cv2.drawKeypoints(img1bgr, kpts1, img1bgr, color=(255,0,0))
    cv2.drawKeypoints(img2bgr, kpts2, img2bgr, color=(0,0,255))
    cv2.imshow(title, np.hstack((img1bgr, img2bgr)))
    cv2.waitKey(1)
    
def draw_img_pair_kpts(img_pair, left_img, right_img, title=""):
    matches = np.array(img_pair["matches"])[img_pair["inliers"]]
    matches_img = cv2.drawMatches(cv2.imread(left_img["img_path"]), left_img["kpts"],
                                  cv2.imread(left_img["img_path"]), right_img["kpts"],
                                  matches,
                                  None
                                  )
                                  
    cv2.imshow(title.format(img_pair["img1_idx"], img_pair["img2_idx"]), matches_img)
    cv2.waitKey(1)
    pass

def detect_keyPts(img):
    orb = cv2.ORB_create(nfeatures=1000)
    keypts, descriptors = orb.detectAndCompute(img, None)
    return keypts, descriptors

def read_images_and_detect_keyPts(idx, DATA_PATH, plot=False):
    """[summary]

    Args:
        idx ([int]): [image pair index]
        DATA_PATH ([string]): path to data
        plot (bool, optional): plot images. Defaults to False.

    Returns:
        img1_dict[dict]: {"idx": idx, "img": img1, "kpts":kpts1, "desc": desc1}
        img2_dict[dict]: {"idx": idx, "img": img2, "kpts":kpts2, "desc": desc2}
    """
    img1, img2, img1path, img2path = read_images(idx, DATA_PATH)
    kpts1, desc1 = detect_keyPts(img1)
    kpts2, desc2 = detect_keyPts(img2)

    img1_dict = {"idx": idx, "img_path": img1path, "kpts":kpts1, "desc": desc1}
    img2_dict = {"idx": idx, "img_path": img2path, "kpts":kpts2, "desc": desc2}


    if plot:
        draw_kpts(img1, img2, kpts1, kpts2, title="kpts[left/right][{}]".format(idx))
        cv2.waitKey(1)

    return img1_dict, img2_dict

def read_cameras(datapath):
    with open(datapath + 'calib.txt') as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]

    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)   
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2

def traingulate_point(P, Q, p, q):
    """ 
    traigulating same points from two cameras with SVD
    Args:  
        P ([float 3x4]): matrix of first camera (K[R1|t1])
        Q ([float 3x4]): matrix of second camera (K[R2|t2])
        p ([float 3x1]): pixel coordinate of point on camera 1
        q ([float 3x1]): pixel coordinate of point on camera 2
    """

    P1, P2, P3 = np.vsplit(P, 3)
    Q1, Q2, Q3 = np.vsplit(Q, 3)

    p1, p2 = [p[0], p[1]]
    q1, q2 = [q[0], q[1]]
    
    A = np.vstack([P3 * p1 - P1,
                   P3 * p2 - P2,
                   Q3 * q1 - Q1,
                   Q3 * q2 - Q2])
    
    U, S, VH = np.linalg.svd(A, full_matrices=True)

    # find min singluar value
    minSingularValueIndex = np.argmin(S)
    minSingularValue = S[minSingularValueIndex]
    X = VH[minSingularValueIndex, ...].reshape(-1, 1)
    X = X/ X[-1]
    return X

def vertical_match_diff(kpts1, kpts2, matches):
    deviations = []

    for ind, match in enumerate(matches):
        ver_diff = np.abs(kpts1[match.queryIdx].pt[1] - kpts2[match.trainIdx].pt[1])
        deviations.append(ver_diff)

    return deviations

def get_matcherScore_inliers_outliers(matches, thres, kpts1, kpts2):
    matches_inliers = []
    matches_outliers =[]
    
    for ind, match in enumerate(matches):
        if match.distance <= thres:
            matches_inliers.append(ind)
        else:
            matches_outliers.append(ind)
        
    return matches_inliers, matches_outliers

def get_rectified_inliers_outliers(kpts1, kpts2, matches, thres):
    match_inliers = []
    match_outliers = []
    
    for ind, match in enumerate(matches):
        ver_diff = np.abs(kpts1[match.queryIdx].pt[1] - kpts2[match.trainIdx].pt[1])
        if ver_diff <= thres:
            match_inliers.append(ind)
        else:
            match_outliers.append(ind)

    return match_inliers, match_outliers
    
def match_images(img_pair, left_img, right_img, matcher="bf", cumsumThres = 0.88, plot=False):
    """[summary]

    Args:
        image_pair (dict): dict containing left and right images
    """

    if matcher == "bf":
        logging.debug("nmatching based on feature BFmatcher distance")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(left_img["desc"],right_img["desc"])

    # sort matches in descending order
    
    matches = sorted(matches, key = lambda x:x.distance)

    if plot:
        # display first 20 points
        imgMatches = cv2.drawMatches(cv2.imread(left_img["img_path"]),left_img["kpts"],right_img["img_path"],right_img["kpts"],matches[:20],None)
        plt.figure(name="Matches")
        plt.imshow(imgMatches)
    
    # rejecting matches based on matcher distance (taking 88% of matches)
    match_dist = []
    for m in matches:
        match_dist.append(m.distance)
        
    counts, bins = plot_histogram(match_dist, plot=plot)
    cumsum_counts = np.cumsum(counts)
    thres = np.argmax(cumsum_counts>cumsumThres)
    max_match_diff = bins[thres]
    matches_inliers, matches_outliers = get_matcherScore_inliers_outliers(matches, thres, left_img["kpts"], right_img["kpts"])
    logging.debug("matcher threshold = {}".format(max_match_diff))
    logging.debug("num of inliers: {}, num of outliers: {}".format(len(matches_inliers), len(matches_outliers)))

    img_pair["matches"] = matches
    img_pair["inliers"] = matches_inliers       
    img_pair["outliers"] = matches_outliers
    img_pair["method"] = "features"
    return img_pair

def match_rectified_images(img_pair, left_img, right_img, feature_matcher="bf", ver_px_diff=2, cumsumThres=0.88, plot=False):
    """ matching two rectified images

    Args:
        img_pair ([dict]): first match based on features, then match
        ver_px_diff (int, optional): Max vertical distance between key points in a rectified image. Defaults to 2.
        cumsumThres (float, optional): percent of sorted matches qualified as "good". Defaults to 0.88.
    """

    img_pair = match_images(img_pair, left_img, right_img, matcher=feature_matcher, cumsumThres=cumsumThres)    # match images according to feature distance
    
    # rejecting outliers that exceed 2 pixels difference:
    inliers, outliers = get_rectified_inliers_outliers(left_img["kpts"], right_img["kpts"], img_pair["matches"], thres=ver_px_diff)
    img_pair["method"] = "features rectified"
    img_pair["inliers"] = inliers

    return img_pair

def plot_histogram(x, plot=False):
    counts, bins = np.histogram(x, bins=list(range(0, 100, 1)))
    counts = counts.astype(np.float)
    counts *= 1 / counts.sum()
    if plot:
        plt.figure(1)
        plt.bar(bins[:-1], counts)
        plt.ylim((0,1))

    return counts, bins

def create_img_pair_from_img_dicts(img1_dict, img2_dict):
    """ generate dict representing image pair

    Args:
        img1_dict (dict): {"idx": idx, "img": img, "kpts":kpts, "desc": desc}
        img2_dict (dict): {"idx": idx, "img": img, "kpts":kpts, "desc": desc}
    
     Returns:
        imgPair (dict): {"img1_idx": img1_dict["idx"],
                        "img2_idx": img2_dict["idx"],
                        "matches": match results
                        "inliers": inds to inliers
                        }
    """

    imgPair = {"img1_idx": img1_dict["idx"],
               "img2_idx": img2_dict["idx"],
               "matches": [],
               "inliers": [],
               "point_cloud": []
    }

    return imgPair

def create_frame_pair_from_img_dicts(img1_dict, img2_dict):
    """ generate dict representing image pair

    Args:
        img1_dict (dict): {"idx": idx, "img": img, "kpts":kpts, "desc": desc}
        img2_dict (dict): {"idx": idx, "img": img, "kpts":kpts, "desc": desc}
    
     Returns:
        imgPair (dict): {"img1_idx": img1_dict["idx"],
                        "img2_idx": img2_dict["idx"],
                        "matches": match results
                        "inliers": inds to inliers
                        }
    """

    imgPair = {"img1_idx": img1_dict["idx"],
               "img2_idx": img2_dict["idx"],
               "matches": [],
               "inliers": [],
               "inliers_frame0": [],
               "inliers_frame1": [],
 
    }

    return imgPair

def generate_point_cloud(img_pair, left_img, right_img, P, Q, plot=False):
    """generate point cloud from a single image pair (left and right cameras)

    Args:
        img_pair (dict): [description]
        P (ndarray): camera matrix left cam
        Q ([type]): camera matrix right cam
        plot (bool, optional): plot point cloud. Defaults to False.
        inliers_idx(int list, optional): if given, only kpts at these match indices are considered

    Returns:
        point_cloud (3 X N array): point in cartesian coordinates
    """
    # traingulating points:
    point_cloud = []
    kpts1kpts2 = np.array([left_img["kpts"][match.queryIdx].pt + right_img["kpts"][match.trainIdx].pt for match in img_pair["matches"]])
    kpts1 = kpts1kpts2[..., :2].transpose()
    kpts2 = kpts1kpts2[..., 2:].transpose()
    x4D = cv2.triangulatePoints(P, Q, kpts1, kpts2)

    point_cloud = x4D[:3, :] / x4D[3, :]
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(point_cloud[0, :], point_cloud[1, :], point_cloud[2, :], alpha=0.5, facecolors="None", color='blue')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
    
    img_pair["point_cloud"] = point_cloud
    return img_pair

def is_inlier(kpt, img_pair, kpType):
    """checks if key point is a match inlier

    Args:
        kpt (int): key point Idx
        img_pair (dict): matches between img pairs
        kpType (string): "qeuryIdx" or "trainIdx
    """
    queryIdx = [match.queryIdx for match in img_pair["matches"]]
    trainIdx = [match.trainIdx for match in img_pair["matches"]]
    if kpType == "queryIdx" :
        if kpt in queryIdx:
            return queryIdx.index(kpt)
        else:
            return -1
    else:
        if kpt in trainIdx:
            return trainIdx.index(kpt)
        else:
            return -1


def get_consistent_matches_between_frames(left_cam_frame_pair, img_pairs, left_imgs, right_imgs):#, left_cam_frame_pair, img_pair):
    """
    clean outliers, by removing key points matches between two frames taken from same camera, 
    but don't have inliers between left and right cameras on same frame.

    Args:
        left_cam_frame_pair [dict]: matches between two frames of same camera
        img_pair [dict]: matches between same frame, taken from two cameras

    Returns:
        left_cam_frame_pair: matches between two frames of left camera
        img_pair: matches with key points also matching on prev frame
    """
    frame0 = left_cam_frame_pair["img1_idx"]
    frame1 = left_cam_frame_pair["img2_idx"]
    frame_matches = left_cam_frame_pair["matches"]
    counter = 0
    while counter < len(left_cam_frame_pair["inliers"]):
        # if object is not a kpt in img oair of current and next frames, remove it
        inlier = left_cam_frame_pair["inliers"][counter]
        inlier_frame0 = is_inlier(left_cam_frame_pair["matches"][inlier].queryIdx, img_pairs[frame0], "queryIdx")
        inlier_frame1 = is_inlier(left_cam_frame_pair["matches"][inlier].trainIdx, img_pairs[frame1], "queryIdx")
        # an inlier must also have match left0 with right0 and left1 with right1.
        if inlier_frame0 >= 0 and inlier_frame1 >= 0:
            left_cam_frame_pair["inliers_frame0"].append(inlier_frame0)
            left_cam_frame_pair["inliers_frame1"].append(inlier_frame1)
            counter += 1
        else:
            left_cam_frame_pair["inliers"].pop(counter)
        
    logging.info("inliers between frame {} and frame {}: {}".format(left_cam_frame_pair["img1_idx"], left_cam_frame_pair["img2_idx"], \
                                                                 len(left_cam_frame_pair["inliers"])))
 
    return left_cam_frame_pair

def get_match_inliers_kpts(img_dict, matches, inliers, kpt_type="queryIdx"):
    if kpt_type=="queryIdx":
        return [img_dict["kpts"][match.queryIdx].pt for match in np.array(matches)[inliers]]
    elif kpt_type=="trainIdx":
        return [img_dict["kpts"][match.trainIdx].pt for match in np.array(matches)[inliers]]


if __name__ == "__main__":
    datapath = "/workspaces/SLAMcourse/VAN_ex/data/dataset05/sequences/05/"
    k, m1, m2 = read_cameras(datapath)
    print("k1=\n{}\n".format(k))
    print("m1=\n{}\n".format(m1))
    print("m2=\n{}\n".format(m2))