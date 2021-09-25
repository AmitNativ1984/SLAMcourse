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
    
def draw_img_pair_kpts(img_pair, title=""):
    matches_img = cv2.drawMatches(cv2.imread(img_pair["img1"]), img_pair["kpts1"],
                                  cv2.imread(img_pair["img2"]), img_pair["kpts2"],
                                  img_pair["inliers"],
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
            matches_inliers.append(match)
        else:
            matches_outliers.append(match)
        
    return matches_inliers, matches_outliers

def get_rectified_inliers_outliers(kpts1, kpts2, matches, thres):
    match_inliers = []
    match_outliers = []
    
    for ind, match in enumerate(matches):
        ver_diff = np.abs(kpts1[match.queryIdx].pt[1] - kpts2[match.trainIdx].pt[1])
        if ver_diff <= thres:
            match_inliers.append(match)
        else:
            match_outliers.append(match)

    return match_inliers, match_outliers
    
def match_images(img_pair, matcher="bf", cumsumThres = 0.88, plot=False):
    """[summary]

    Args:
        image_pair (dict): dict containing left and right images
    """

    if matcher == "bf":
        logging.debug("nmatching based on feature BFmatcher distance")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(img_pair["desc1"],img_pair["desc2"])

    # sort matches in descending order
    
    matches = sorted(matches, key = lambda x:x.distance)

    if plot:
        # display first 20 points
        imgMatches = cv2.drawMatches(img_pair["img1"],img_pair["kpts1"],img_pair["img2"],img_pair["kpts2"],matches[:20],None)
        cv2.imshow("Matches", imgMatches)
    
    # rejecting matches based on matcher distance (taking 88% of matches)
    match_dist = []
    for m in matches:
        match_dist.append(m.distance)
        
    counts, bins = plot_histogram(match_dist, plot=plot)
    cumsum_counts = np.cumsum(counts)
    thres = np.argmax(cumsum_counts>cumsumThres)
    max_match_diff = bins[thres]
    matches_inliers, matches_outliers = get_matcherScore_inliers_outliers(matches, thres, img_pair["kpts1"], img_pair["kpts2"])
    logging.debug("matcher threshold = {}".format(max_match_diff))
    logging.debug("num of inliers: {}, num of outliers: {}".format(len(matches_inliers), len(matches_outliers)))

    img_pair["inliers"] = matches_inliers       
    img_pair["outliers"] = matches_outliers
    img_pair["method"] = "features"
    return img_pair

def match_rectified_images(img_pair, feature_matcher="bf", ver_px_diff=2, cumsumThres=0.88, plot=False):
    """ matching two rectified images

    Args:
        img_pair ([dict]): first match based on features, then match
        ver_px_diff (int, optional): Max vertical distance between key points in a rectified image. Defaults to 2.
        cumsumThres (float, optional): percent of sorted matches qualified as "good". Defaults to 0.88.
    """

    img_pair = match_images(img_pair, matcher=feature_matcher, cumsumThres=cumsumThres)    # match images according to feature distance
    
    # rejecting outliers that exceed 2 pixels difference:
    inliers, outliers = get_rectified_inliers_outliers(img_pair["kpts1"], img_pair["kpts2"], img_pair["inliers"], thres=ver_px_diff)
    img_pair["method"] = "features rectified"
    img_pair["inliers"] = inliers
    img_pair["outliers"] += outliers

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
                        "kpts1": img1_dict["kpts"],
                        "desc1": img1_dict["desc"],
                        "kpts2": img2_dict["kpts"],
                        "desc2": img2_dict["desc"],
                        "img1": img1_dict["img_path"],
                        "img2": img2_dict["img_path"]}
    
    """

    imgPair = (
                {"img1_idx": img1_dict["idx"],
                "img2_idx": img2_dict["idx"],
                "kpts1": img1_dict["kpts"],
                "desc1": img1_dict["desc"],
                "kpts2": img2_dict["kpts"],
                "desc2": img2_dict["desc"],
                "img1": img1_dict["img_path"],
                "img2": img2_dict["img_path"],
                "inliers": [],
                "outliers": []}
    )

    return imgPair

def generate_point_cloud(img_pair, P, Q, inliers_idx=None, plot=False):
    """generate point cloud from a single image pair (left and right cameras)

    Args:
        img_pair (dict): [description]
        P (ndarray): camera matrix left cam
        Q ([type]): camera matrix right cam
        plot (bool, optional): plot point cloud. Defaults to False.

    Returns:
        point_cloud (3 X N array): point in cartesian coordinates
    """
    # traingulating points:
    point_cloud = []
    if inliers_idx is None:
        kpts1kpts2 = np.array([img_pair["kpts1"][match.queryIdx].pt + img_pair["kpts2"][match.trainIdx].pt for match in img_pair["inliers"]])
    else:
        kpts1kpts2 = np.array([img_pair["kpts1"][match.queryIdx].pt + img_pair["kpts2"][match.trainIdx].pt for idx, match in enumerate(img_pair["inliers"]) if idx in inliers_idx])
    
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
    
    return point_cloud

def get_consistent_matches_between_frames(frame_seq, img_pair):
    """
    clean outliers, by removing key points matches between two frames taken from same camera, 
    but don't have inliers between left and right cameras on same frame.

    Args:
        frame_seq [dict]: matches between two frames of same camera
        img_pair [dict]: matches between same frame, taken from two cameras

    Returns:
        frame_seq: matches between two frames of left camera
        img_pair: matches with key points also matching on prev frame
    """

    frame_seq_queryIdx = [match.queryIdx for match in frame_seq["inliers"]]
    frame_seq_trainIdx = [match.trainIdx for match in frame_seq["inliers"]]

    img_pair2_queryIdx  = [match.queryIdx for match in img_pair[frame_seq["img2_idx"]]["inliers"]]

    inliers_frame1 = []
    inliers_frame2 = []
    inliers_between_frames = []
    for img_pair1_match_idx, img_pair1_match in enumerate(img_pair[frame_seq["img1_idx"]]["inliers"]):
        # does kpts1 in this image pair also exists in match between frames?
        try:
            match_between_frames_idx = frame_seq_queryIdx.index(img_pair1_match.queryIdx)
        except Exception: 
            img_pair[frame_seq["img1_idx"]]["outliers"].append(img_pair1_match)
            continue

        fram0_img1_kpt = img_pair1_match.queryIdx
        # what is kpt index on the ?
        try:
            img_pair2_inlier_idx = img_pair2_queryIdx.index(frame_seq["inliers"][match_between_frames_idx].trainIdx)
        
        except Exception: 
            continue
        
        inliers_frame1.append(img_pair[frame_seq["img1_idx"]]["inliers"][img_pair1_match_idx])
        inliers_frame2.append(img_pair[frame_seq["img2_idx"]]["inliers"][img_pair2_inlier_idx])
        inliers_between_frames.append(frame_seq["inliers"][match_between_frames_idx])
                
    img_pair[frame_seq["img1_idx"]]["inliers"] = inliers_frame1
    img_pair[frame_seq["img2_idx"]]["inliers"] = inliers_frame2
    frame_seq["inliers"] = inliers_between_frames

    logging.info("inliers frame {}: {}; inliers frame {}: {}".format(frame_seq["img1_idx"], len(img_pair[frame_seq["img1_idx"]]["inliers"]),
                                                                 frame_seq["img2_idx"], len(img_pair[frame_seq["img2_idx"]]["inliers"])))

 
    return frame_seq, img_pair

if __name__ == "__main__":
    datapath = "/workspaces/SLAMcourse/VAN_ex/data/dataset05/sequences/05/"
    k, m1, m2 = read_cameras(datapath)
    print("k1=\n{}\n".format(k))
    print("m1=\n{}\n".format(m1))
    print("m2=\n{}\n".format(m2))