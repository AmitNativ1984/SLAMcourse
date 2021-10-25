from operator import truediv
import numpy as np
import random
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

def draw_kpts(img1, img2, kpts1, kpts2, title="kpts[left/right]", plot=True, color=(0, 0, 255)):
    if img1.shape[-1] != 3:
        img1bgr = np.dstack((img1, img1, img1))
    else:
        img1bgr = img1

    if img2.shape[-1] != 3:
        img2bgr = np.dstack((img2, img2, img2))
    else:
        img2bgr = img2

    cv2.drawKeypoints(img1bgr, kpts1, img1bgr, color=color)
    cv2.drawKeypoints(img2bgr, kpts2, img2bgr, color=color)
    img_stacked = np.hstack((img1bgr, img2bgr))
    if plot:
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, img_stacked)
        # cv2.resizeWindow(title, 2200, 2200)
        cv2.waitKey(1)

    return img_stacked, img1bgr, img2bgr
    
def draw_img_pair_kpts(img_pair, left_img, right_img, title="", inliers=None):
   
    matches = np.array(img_pair["matches"])[img_pair["inliers"]]
    if inliers is not None:
        matches = matches[inliers]

    matches_img = cv2.drawMatches(cv2.imread(left_img["img_path"]), left_img["kpts"],
                                  cv2.imread(right_img["img_path"]), right_img["kpts"],
                                  matches[:4],
                                  cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                  )
                                  
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title.format(img_pair["img1_idx"], img_pair["img2_idx"]), matches_img)
    # cv2.resizeWindow(title, 2600, 2600)
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
    
def match_images(img_pair, left_img, right_img, matcher="bf", cumsumThres = 0.88, plot=False, title=""):
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

    # if plot:
    #     # display first 20 points
    #     imgMatches = cv2.drawMatches(cv2.imread(left_img["img_path"]),left_img["kpts"],cv2.imread(right_img["img_path"]),right_img["kpts"],matches[:20],None)
    #     plt.figure(name="Matches")
    #     plt.imshow(imgMatches)

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
    
    if plot:
        imgMatches = cv2.drawMatches(cv2.imread(left_img["img_path"]),left_img["kpts"],cv2.imread(right_img["img_path"]),right_img["kpts"],matches[:20],None)
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, imgMatches)
        cv2.resizeWindow(title, 2240, 1280)
        cv2.waitKey(1)
    
    return img_pair

def match_rectified_images(img_pair, left_img, right_img, feature_matcher="bf", ver_px_diff=2, cumsumThres=0.88, plot=False, title=""):
    """ matching two rectified images

    Args:
        img_pair ([dict]): first match based on features, then match
        ver_px_diff (int, optional): Max vertical distance between key points in a rectified image. Defaults to 2.
        cumsumThres (float, optional): percent of sorted matches qualified as "good". Defaults to 0.88.
    """

    img_pair = match_images(img_pair, left_img, right_img, matcher=feature_matcher, cumsumThres=cumsumThres, plot=plot, title=title + "bf_match")    # match images according to feature distance
    
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

def get_inliers_px(img_dict, matches, inliers, kpt_type="queryIdx"):
    if kpt_type=="queryIdx":
        return [img_dict["kpts"][match.queryIdx].pt for match in np.array(matches)[inliers]]
    elif kpt_type=="trainIdx":
        return [img_dict["kpts"][match.trainIdx].pt for match in np.array(matches)[inliers]]

def get_kpts_from_match_inliers(img_dict, matches, inliers, kpt_type="queryIdx"):
    if kpt_type=="queryIdx":
        return [img_dict["kpts"][match.queryIdx] for match in np.array(matches)[inliers]]
    elif kpt_type=="trainIdx":
        return [img_dict["kpts"][match.trainIdx] for match in np.array(matches)[inliers]]

def px_dist_euclidian(px1, px2):
    """calculate euclidian distantce between arrays of pixels

    Args:
        px1 (2XN numpy array): pixels in img1
        px2 (2XN numpy array): pixels in img2
    """
    A = (px1 - px2)
    return np.sqrt(np.sum(A**2, axis=0))

def project_world_coord_to_image_pixels(P, X):
    """project 3D world homegenous coordinate coordinate to image pixels 

    Args:
        P ([Mat]): [projection matrix]
        X ([Mat]): [world coordinates]

    Returns:
        [Mat]: [pixelsv projected rto camera plane]
    """
    proj_hom = P @ X
    proj_px = proj_hom[:2, :] / proj_hom[-1, :]

    return proj_px

def get_supporters(left_cam_pair, img_pairs, left_imgs, right_imgs, T, K, M1, M2, thres=2):
    """Get all supporters of transform T (a projection transform of 3D point from camera A coordinate system to camera B)

    Args:
        left_cam_frame_pair (dict): pair of left0 and left1, including their inliers
        img_pairs (dict): dict including inliers of each left-right pair
        left_imgs (array[dict]): left images key points and desc
        right_imgs (array[dict]): right images key points and desc
        T (3x4 ndarray): projection transfrom from left0 to left1 coordinate system
        K (3x4 ndarray): projection of camera intrinsic parameters
        M (3x4 ndarray): translation from left to right camera
        thres (int, optional): minimal pixel distance between projection of point and its original key point. Defaults to 2.
    """
    img_pair_frame0 = img_pairs[left_cam_pair["img1_idx"]]
    img_pair_frame1 = img_pairs[left_cam_pair["img2_idx"]]
    inliers_frame0 = np.array(left_cam_pair["inliers_frame0"])
    inliers_frame1 = np.array(left_cam_pair["inliers_frame1"])

    world_points = img_pair_frame0["point_cloud"][..., inliers_frame0]

    px_left0 = get_inliers_px(img_dict=left_imgs[img_pair_frame0["img1_idx"]],
                                matches=img_pair_frame0["matches"], 
                                inliers=inliers_frame0, 
                                kpt_type="queryIdx")

    px_right0 = get_inliers_px(img_dict=right_imgs[img_pair_frame0["img2_idx"]],
                                matches=img_pair_frame0["matches"], 
                                inliers=inliers_frame0, 
                                kpt_type="trainIdx")
    
    px_left1 = get_inliers_px(img_dict=left_imgs[img_pair_frame1["img1_idx"]],
                                matches=img_pair_frame1["matches"], 
                                inliers=inliers_frame1, 
                                kpt_type="queryIdx")

    px_right1 = get_inliers_px(img_dict=right_imgs[img_pair_frame1["img2_idx"]],
                                matches=img_pair_frame1["matches"], 
                                inliers=inliers_frame1, 
                                kpt_type="trainIdx")
    
    ## Project world points of matching inliers back to images, using calculated transfrom from pnp
    world_points = img_pair_frame0["point_cloud"][:, left_cam_pair["inliers_frame0"]] # 3D point that is matched over the two image pairs:
    X = np.vstack((world_points, np.ones((1, world_points.shape[1]))))
    proj_px_left0 = project_world_coord_to_image_pixels(K @ M1, X)
    proj_px_right0 = project_world_coord_to_image_pixels(K @ M2, X)
    proj_px_left1 = project_world_coord_to_image_pixels(K @ M1 @ T, X)
    proj_px_right1 = project_world_coord_to_image_pixels(K @ M2 @ T, X)
        

    # calculate distance between true pixels
    dist_left1 = px_dist_euclidian(proj_px_left0, np.array(px_left0).T)
    dist_right1 = px_dist_euclidian(proj_px_right0,  np.array(px_right0).T)
    dist_left2 = px_dist_euclidian(proj_px_left1,  np.array(px_left1).T)
    dist_right2 = px_dist_euclidian(proj_px_right1,  np.array(px_right1).T)

    ind = (dist_left1 <= thres) & \
          (dist_right1 <= thres) & \
          (dist_left2 <= thres) & \
          (dist_right2 <= thres)    

    supporters = np.array(list(range(len(dist_left1))))[ind]

    return supporters

def pnp_transform(left_cam_pair, img_pairs, left_imgs, K, inliers_inds):
    """
    1. sample 4 points [V]
    2. plot the 4 key points on all 4 images (red) [V]
    3. perform pnp and get transform T [V]
    4. project matching 3D points back to left0, right0, left1, right1, and plot pixels in red   
    """

    img_pair_frame0 = img_pairs[left_cam_pair["img1_idx"]]
    img_pair_frame1 = img_pairs[left_cam_pair["img2_idx"]]
    inliers_frame0 = np.array(left_cam_pair["inliers_frame0"])[inliers_inds]
    inliers_frame1 = np.array(left_cam_pair["inliers_frame1"])[inliers_inds]

    ## calculate PnP:
    world_points = img_pair_frame0["point_cloud"][..., inliers_frame0]
    image_points = get_inliers_px(img_dict=left_imgs[img_pair_frame1["img1_idx"]],
                                            matches=img_pair_frame1["matches"], 
                                            inliers=inliers_frame1, 
                                            kpt_type="queryIdx")
    
    logging.info("3D world coordinates from frame 0:")
    logging.info(world_points)
    logging.info("using frame0 points to as TRUE 3D position for PnP calculation on frame 1")


    # calculate PnP:
    success, R1, t1 = cv2.solvePnP(objectPoints=world_points.transpose(), 
                                   imagePoints=np.array(image_points),
                                   cameraMatrix=K[:,:3], 
                                   distCoeffs=np.zeros((4,1)),
                                   flags=cv2.SOLVEPNP_P3P
    )
   
    if success == False:
        logging.warning("PnP failed")
        return success, []
    
    R1, _ = cv2.Rodrigues(R1)
    logging.info("PnP succeeded")

    ## calculate T transform:
    T = np.hstack((R1, t1))
    I = np.array([0, 0, 0, 1])
    T = np.vstack((T, I))

    logging.info("frame left {} PnP solution: R={}".format(1, R1))
    logging.info("frame left {} PnP solution: t={}".format(1, t1.transpose()))

    return success, T

    ## Transform projected selected points back to image:
    # X = np.vstack((world_points, np.ones((1, world_points.shape[1]))))

    # # transform all 3D world points to left2 coordinate system:
    

    # plot the 4 keypoints on all 4 images (left0/right0; left1/right1)
    # left0 = cv2.imread(left_imgs[left_cam_pair["img1_idx"]]["img_path"])
    # right0 = cv2.imread(right_imgs[left_cam_pair["img1_idx"]]["img_path"])
    # left1 = cv2.imread(left_imgs[left_cam_pair["img2_idx"]]["img_path"])
    # right1 = cv2.imread(right_imgs[left_cam_pair["img2_idx"]]["img_path"])

    # left1kpts = get_kpts_from_match_inliers(img_dict=left_imgs[left_cam_pair["img1_idx"]],
    #                                     matches=img_pair_frame0["matches"],
    #                                     inliers=inliers_frame0,
    #                                     kpt_type="queryIdx")

    # right1kpts = get_kpts_from_match_inliers(img_dict=right_imgs[left_cam_pair["img1_idx"]],
    #                                     matches=img_pair_frame0["matches"],
    #                                     inliers=inliers_frame0,
    #                                     kpt_type="trainIdx")

    # left2kpts = get_kpts_from_match_inliers(img_dict=left_imgs[left_cam_pair["img2_idx"]],
    #                                     matches=img_pair_frame1["matches"],
    #                                     inliers=inliers_frame1,
    #                                     kpt_type="queryIdx")

    # right2kpts = get_kpts_from_match_inliers(img_dict=right_imgs[left_cam_pair["img2_idx"]],
    #                                     matches=img_pair_frame1["matches"],
    #                                     inliers=inliers_frame1,
    #                                     kpt_type="trainIdx")

    
    # for px_left0, px_right0, px_left1, px_right1 in zip(proj_px_left0.T, proj_px_right0.T, proj_px_left1.T, proj_px_right1.T):
    #     left0 = cv2.circle(left0, tuple(px_left0.astype(int)), radius=5, color=(255, 255, 0), thickness=1)
    #     right0 = cv2.circle(right0, tuple(px_right0.astype(int)), radius=5, color=(255, 255, 0), thickness=1)
    #     left1 = cv2.circle(left1, tuple(px_left1.astype(int)), radius=5, color=(255, 255, 0), thickness=1)
    #     right1 = cv2.circle(right1, tuple(px_right1.astype(int)), radius=5, color=(255, 255, 0), thickness=1)

    # left0right0 = draw_kpts(img1 = left0, 
    #                         img2 = right0,
    #                         kpts1=left1kpts,
    #                         kpts2=right1kpts,
    #                         plot=False
    # )
    # left1right1 = draw_kpts(img1 = left1, 
    #                         img2 = right1,
    #                         kpts1=left2kpts,
    #                         kpts2=right2kpts,
    #                         plot=False
    # )
    
    # left0right0_left1right1 = np.vstack((left0right0, left1right1))
    # title="point for pnp calc [top: frame0, bottom: frame1]"
    # cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    # cv2.imshow(title, left0right0_left1right1)
    # cv2.waitKey(1)
    # cv2.waitKey(0)

if __name__ == "__main__":
    datapath = "/workspaces/SLAMcourse/VAN_ex/data/dataset05/sequences/05/"
    k, m1, m2 = read_cameras(datapath)
    print("k1=\n{}\n".format(k))
    print("m1=\n{}\n".format(m1))
    print("m2=\n{}\n".format(m2))