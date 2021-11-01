import os
from utils import *
import matplotlib.pyplot as plt
import random

DATA_PATH = r'../data/dataset05/sequences/05/'

if __name__ == "__main__":
    """ [2.1] match image pair #1 """
    print("\n")
    logging.info("Begin EX 2")
    img_pairs=[]
    # get camera matrices
    K, M1, M2 = read_cameras(DATA_PATH)
    P = K @ M1
    Q = K @ M2
    logging.info("finished loading Kitti camera calibrations")
    """
    imgPair = (
                {"left_img_idx": img1_dict["idx"],
                "right_img_idx": img2_dict["idx"],
                "method": "features",
                "matches": match results
                "inliers": inds to inliers 
                "point_cloud": 3D point cloud of all matches
                } 
    )
    """
    print("\n")
    logging.info("==== Answer [2.1]:=====")
    """
        Use the code from exercise 1 to create a point cloud for the next images pair (i.e. match
        features, remove outliers and triangulate):
    """
    point_cloud = []
    left_imgs = []
    right_imgs = []
    for idx in range(0,2):
        img1_dict, img2_dict = read_images_and_detect_keyPts(idx, DATA_PATH, plot=False)
        left_imgs.append(img1_dict)
        right_imgs.append(img2_dict)
        img_pairs.append(create_img_pair_from_img_dicts(left_imgs[idx], right_imgs[idx]))
        # match key points
        img_pairs[idx] = match_rectified_images(img_pairs[idx], left_imgs[idx], right_imgs[idx], feature_matcher="bf", cumsumThres = 0.88, plot=False, title="img pair:[left {}, right {}]".format(idx, idx))
        draw_img_pair_kpts(img_pairs[idx], left_imgs[idx], right_imgs[idx], title="img pair:[left {}, right {}]".format(idx, idx))

        # get point cloud
        img_pairs[idx] = generate_point_cloud(img_pairs[idx], left_imgs[idx], right_imgs[idx], P, Q, plot=False)
        logging.info("calculated 3D point cloud for image pair {}".format(idx))

    
    # match points in two left images
    print("\n")
    logging.info("===== Answer [2.2] =====")
    """
        Match features between the two left images
    """
    left_cam_pairs = []
    for idx in range(0, 1):
        left_cam_pairs.append(create_frame_pair_from_img_dicts(left_imgs[idx], left_imgs[idx + 1])) 
        left_cam_pairs[idx] = match_images(left_cam_pairs[idx], left_imgs[idx], left_imgs[idx+1], matcher="bf", cumsumThres=0.88, plot=False)
        left_cam_pairs[idx] = get_consistent_matches_between_frames(left_cam_pairs[idx], img_pairs, left_imgs, right_imgs)
        draw_img_pair_kpts(left_cam_pairs[idx], left_imgs[idx], left_imgs[idx+1], title="frame pair:[left {}, left {}]".format(idx, idx+1))
        logging.info("matched features in frame seq: left[{}], left[{}]".format(idx, idx+1))
    
    print("\n")
    logging.info("===== Answer [2.3] =====")
    """
        (*) Define [R|t] a transformation T that transofrms from left0 coordinate system to left1:
            1. We have 3D object points in left0 coordinate systems.
            2. We have the matching pixels in left1 camera.
            3. Use cv2.solvePnP to get [R|t] - the transformation of 3D points in left0 coordinates to left1 coordinates.
        (*) Camera A hase extrinsics [I|0]. T(A->B) = R1x + t1; the transofrmation of 3D points to camera B coordinates.
            T(B->C) = R2x + t2; the trasnformation from B to camera C coordinate system. What is T(A->C)?
            
            1. T(A->C) = T(B->C)[T(A->B)] = R2(R1x + t1) + t2 = R2R1x + R2t1 + t2 = [R2R1 | (R2t1 + t2)]
        
        (*) For a camera with extrinsic matrix [R|t], what is the location of the camera in the global
            coordinate system?

            1. 0 = Rx + t, so that Xw = -R't
    """

    # finding kpts that were matches between left0, right1, left1, right1.
    # thest are matches that are found in: inliers[left0, right0]; inliers[left0, left1]; inliers[left1, right1]

    # randomly select 4 key-points that were matched on all four images (2 image pairs from frame0 and frame 1)
    idx = 0
    # building point cloud only from key points that are matched on all four images (2 image pairs from frame0 and frame1)
    success = False
    while not success:
        inds = np.array(random.sample(range(len(left_cam_pairs[idx]["inliers"])), 4))
        inliers = np.array(left_cam_pairs[idx]["inliers"])[inds]
        img_pair_frame0 = img_pairs[left_cam_pairs[idx]["img1_idx"]]
        img_pair_frame1 = img_pairs[left_cam_pairs[idx]["img2_idx"]]
        inliers_frame0 = np.array(left_cam_pairs[idx]["inliers_frame0"])[inds]
        inliers_frame1 = np.array(left_cam_pairs[idx]["inliers_frame1"])[inds]
    
        # object world points as point cloud of frame 0
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
                                        cameraMatrix=M1[:,:3], 
                                        distCoeffs=np.zeros((4,1)),
                                        flags=cv2.SOLVEPNP_P3P
        )
        try:
            R1, jacobian = cv2.Rodrigues(R1)
            success=True
            logging.info("PnP succeeded")
        except Exception as exception:
            logging.info("PnP failed")
            logging.warning(exception)
            success=False



    # Xc = RXw + t = [R | t]
    # to find camera position, we have to find Xw such that Xc = 0
    # Xw = inv(R)Xc - inv(R)t = [inv(R) | -inv(R)t]
    # Because R is unitary: inv(R) = R':
    # Xw = R'Xc - R't = [R' | -R't]
    
    T = []
    T.append(np.hstack((R1, -R1 @ t1)))

    logging.info("frame left {} PnP solution: R={}".format(1, R1))
    logging.info("frame left {} PnP solution: t={}".format(1, t1.transpose()))

    camPos_left = []
    camRot_left = []
    camPos_right = []
    camRot_right = []
    yaw_pitch_roll = []
    
    camPos_left.append(np.zeros((3,1)))
    camRot_left.append(M1[:, :3])
    
    camPos_left.append(-R1.transpose() @ t1)
    camRot_left.append(R1.transpose())
    # get camera rotation:
    yaw_pitch_roll.append(cv2.decomposeProjectionMatrix(projMatrix=np.hstack((camRot_left[0], camPos_left[0])), cameraMatrix=K.copy())[6])

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    for idx in range(0, 2):
        # calculating position of right camera
        camPos_right.append(camPos_left[idx] - M2[:, -1].reshape(-1, 1))
        camRot_right.append(camRot_left[idx])
        yaw_pitch_roll.append(cv2.decomposeProjectionMatrix(projMatrix=np.hstack((camRot_left[idx], camPos_left[idx])), cameraMatrix=K.copy())[6])
        logging.info("frame {} LEFT:  cam Pos={}[m]; yaw={}[deg], pitch={}[deg], roll={}[deg]".format(idx, camPos_left[idx].transpose()[0], yaw_pitch_roll[idx][0], yaw_pitch_roll[0][1], yaw_pitch_roll[idx][2]))
        logging.info("frame {} RIGHT: cam Pos={}[m]; yaw={}[deg], pitch={}[deg], roll={}[deg]".format(idx, camPos_right[idx].transpose()[0], yaw_pitch_roll[idx][0], yaw_pitch_roll[idx][1], yaw_pitch_roll[idx][2]))
        plt.scatter(camPos_left[idx][0], camPos_left[idx][2], marker='o', alpha=0.5, facecolors="None", color='blue')
        plt.scatter(camPos_right[idx][0], camPos_right[idx][2], marker='^', alpha=0.5, facecolors="None", color='blue')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    # ax.set_zlabel('Z [m]')

    print("\n")
    logging.info("===== Answer [2.4] =====")    
    """Finding supporters of transform T. Here are the guidlines:
        1. Get projection matrix for every frame, from the calculations above: Pleft0, Pright0, Pleft1, Pright1
        2. For every 3D world that has matches on all 4 frames:
            2.1 Find its pixels from key point location on each frame.
            2.2 Project 3D world poistion to camera.
            2.3 If projection is up to 2 pixels from key point position, this is a supporter
    """
    inds = np.array(range(len(left_cam_pairs[0]["inliers"])))
    num_supporters = 0
    while len(inds) > 4:
        np.random.shuffle(inds)
        inliers_inds = inds[:4]
        inds = inds[4:]
        success, t = pnp_transform(left_cam_pairs[0], img_pairs, left_imgs, K, inliers_inds)
        if not success:
            continue

        supp_inds = get_supporters(left_cam_pairs[0], img_pairs=img_pairs, left_imgs=left_imgs, right_imgs=right_imgs, T=t, K=K, M1=M1, M2=M2, thres=2)
        if len(supp_inds) > num_supporters:
            T = t
            supporters_inds = supp_inds
            num_supporters = len(supporters_inds)
        break

    left_cam_pairs[0]["pnp_supporters"] = supporters_inds
    # plot matches on left0 and left1 in red in supporters and supporters in cyan:
    
    inliers_frame0 = np.array(left_cam_pairs[0]["inliers_frame0"])
    inliers_frame1 = np.array(left_cam_pairs[0]["inliers_frame1"])
    
    
    left0kpts = get_kpts_from_match_inliers(img_dict=left_imgs[left_cam_pairs[0]["img1_idx"]],
                                            matches=img_pairs[left_cam_pairs[0]["img1_idx"]]["matches"],
                                            inliers=inliers_frame0,
                                            kpt_type="queryIdx"
    )
    
    left1kpts = get_kpts_from_match_inliers(img_dict=left_imgs[left_cam_pairs[0]["img2_idx"]],
                                            matches=img_pairs[left_cam_pairs[0]["img2_idx"]]["matches"],
                                            inliers=inliers_frame1,
                                            kpt_type="queryIdx")

    left0supp = get_kpts_from_match_inliers(img_dict=left_imgs[left_cam_pairs[0]["img1_idx"]],
                                            matches=img_pairs[left_cam_pairs[0]["img1_idx"]]["matches"],
                                            inliers=inliers_frame0[supporters_inds],
                                            kpt_type="queryIdx"
    )
    
    
    left1supp = get_kpts_from_match_inliers(img_dict=left_imgs[left_cam_pairs[0]["img2_idx"]],
                                            matches=img_pairs[left_cam_pairs[0]["img2_idx"]]["matches"],
                                            inliers=inliers_frame1[supporters_inds],
                                            kpt_type="queryIdx")

    
    left0=cv2.imread(left_imgs[left_cam_pairs[0]["img1_idx"]]["img_path"])
    left1=cv2.imread(left_imgs[left_cam_pairs[0]["img2_idx"]]["img_path"])
    
    _, left0, left1 = draw_kpts(left0, 
                            left1,
                            kpts1=left0kpts,
                            kpts2=left1kpts,
                            plot=False
    )

    left0left1, left0, left1 = draw_kpts(left0, 
                                        left1,
                                        kpts1=left0supp,
                                        kpts2=left1supp,
                                        plot=False,
                                        color=(255, 255, 0)
    )

    title="ransac supporters step 0 [left0 | left1] = {}".format(len(supporters_inds))
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, left0left1)


    logging.info("===== Answer [2.5] =====")
    """
        Use a RANSAC framework, with PNP as the inner model, to find the 4 points that maximize
        the number of supporters. We call this maximal group the ‘inliers’.
        Note: Implement RANSAC yourself, do not use ‘cv2.solvePnPRansac’
        Refine the resulting transformation by calculating transformation T for all the inliers.
    """
    
    ransac_pnp = RANSAC_PNP(K=K,
                            M1=M1,
                            M2=M2
    )
    
    left_cam_pairs = ransac_pnp.initial_supporters(left_cam_pairs, 
                                                    img_pairs, 
                                                    left_imgs, 
                                                    right_imgs, 
                                                    K, 
                                                    M1, 
                                                    M2, 
                                                    pair=0, 
                                                    num_points=4, 
                                                    thres=2
    )

    left_cam_pairs, T = ransac_pnp.refinement_stage(left_cam_pairs, 
                                                    img_pairs, 
                                                    left_imgs, 
                                                    right_imgs, 
                                                    K, 
                                                    M1, 
                                                    M2, 
                                                    pair=0, 
                                                    num_points=4, 
                                                    thres=2
    )
      
    inliers_frame0 = np.array(left_cam_pairs[0]["inliers_frame0"])
    inliers_frame1 = np.array(left_cam_pairs[0]["inliers_frame1"])
    
    left0in = get_kpts_from_match_inliers(img_dict=left_imgs[left_cam_pairs[0]["img1_idx"]],
                                            matches=img_pairs[left_cam_pairs[0]["img1_idx"]]["matches"],
                                            inliers=inliers_frame0,
                                            kpt_type="queryIdx"
    )
    
    left1in = get_kpts_from_match_inliers(img_dict=left_imgs[left_cam_pairs[0]["img2_idx"]],
                                            matches=img_pairs[left_cam_pairs[0]["img2_idx"]]["matches"],
                                            inliers=inliers_frame1,
                                            kpt_type="queryIdx")

    

    
    left0=cv2.imread(left_imgs[left_cam_pairs[0]["img1_idx"]]["img_path"])
    left1=cv2.imread(left_imgs[left_cam_pairs[0]["img2_idx"]]["img_path"])
    
    _, left0, left1 = draw_kpts(left0, 
                            left1,
                            kpts1=left0kpts,
                            kpts2=left1kpts,
                            plot=False,
                            color=(255, 255, 0)
    )

    left0left1, left0, left1 = draw_kpts(left0, 
                                        left1,
                                        kpts1=left0in,
                                        kpts2=left1in,
                                        plot=False,
                                        color=(0, 0, 255)
    )
    title="ransac supporters refinement [left0 | left1] = {} red: inliers; cyan: outliers".format(len(inliers_frame0))
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, left0left1)
    

    # plotting points clouds of frame 0 and frame1
    point_cloud_frame0 = img_pairs[0]["point_cloud"][..., inliers_frame0]
    point_cloud_frame1 = img_pairs[1]["point_cloud"][..., inliers_frame1]

    # transform frame0 point cloud to frame1 coordinates:
    point_cloud_frame0 = np.vstack((point_cloud_frame0, np.ones((1, point_cloud_frame0.shape[1]))))
    transformed_cloud_frame0 = T @ point_cloud_frame0
    transformed_cloud_frame0 = transformed_cloud_frame0[:3, :]
    pclfig = plt.figure()
    ax = pclfig.add_subplot()#(projection='3d')

    point_cloud_frame1 = img_pairs[1]["point_cloud"][..., inliers_frame1]
    ax.scatter(transformed_cloud_frame0[0, :], transformed_cloud_frame0[2, :], s=25**2, alpha=0.2, color='blue', label='cloud0 trans')
    ax.scatter(point_cloud_frame1[0, :], point_cloud_frame1[2, :], s=25**2,alpha=0.2, color='red', marker="s", label='cloud1')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.legend()
    ax.grid()
    # ax.set_zlabel('Z [m]')
    
    
    cv2.waitKey(0)
    plt.show()