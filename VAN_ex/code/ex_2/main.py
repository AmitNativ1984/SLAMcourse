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
    point_cloud = []
    left_imgs = []
    right_imgs = []
    for idx in range(0,2):
        img1_dict, img2_dict = read_images_and_detect_keyPts(idx, DATA_PATH, plot=False)
        left_imgs.append(img1_dict)
        right_imgs.append(img2_dict)
        img_pairs.append(create_img_pair_from_img_dicts(left_imgs[idx], right_imgs[idx]))
        # match key points
        img_pairs[idx] = match_rectified_images(img_pairs[idx], left_imgs[idx], right_imgs[idx], feature_matcher="bf", cumsumThres = 0.88, plot=False)
        draw_img_pair_kpts(img_pairs[idx], left_imgs[idx], right_imgs[idx], title="img pair:[left {}, right {}]")

        # get point cloud
        img_pairs[idx] = generate_point_cloud(img_pairs[idx], left_imgs[idx], right_imgs[idx], P, Q, plot=True)
        logging.info("calculated 3D point cloud for image pair {}".format(idx))

    
    # match points in two left images
    print("\n")
    logging.info("===== Answer [2.2] =====")
    left_cam_pairs = []
    for idx in range(0, 1):
        left_cam_pairs.append(create_frame_pair_from_img_dicts(left_imgs[idx], left_imgs[idx + 1])) 
        left_cam_pairs[idx] = match_images(left_cam_pairs[idx], left_imgs[idx], left_imgs[idx+1], matcher="bf", cumsumThres=0.88, plot=False)
        left_cam_pairs[idx] = get_consistent_matches_between_frames(left_cam_pairs[idx], img_pairs, left_imgs, right_imgs)
        draw_img_pair_kpts(left_cam_pairs[idx], left_imgs[idx], left_imgs[idx+1], title="frame pair:[left {}, left {}]")
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
            
            1. T(A->C) = T(B->C)[T(A->B)] = R2(R1x + t1) + t2 = R2R1x + R2t1 + t1 = [R2R1 | (R2 + I)t1]
        
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
        match_between_frames_idx = random.sample(range(len(left_cam_pairs[idx]["inliers"])), 4)
        
        left0_kpt_ind = []
        left1_kpt_ind = []
        for match_idx in match_between_frames_idx:
            left0_kpt_ind.append(is_inlier(left_cam_pairs[idx]["inliers"][match_idx].queryIdx, img_pairs[idx], "queryIdx")[1])
            left1_kpt_ind.append(is_inlier(left_cam_pairs[idx]["inliers"][match_idx].trainIdx, img_pairs[idx+1], "queryIdx")[1])
        
            frame0


        logging.info("3D world coordinates from frame 0:")
        logging.info(world_points)
        logging.info("using frame0 points to as TRUE 3D position for PnP calculation on frame 1")
        
        image_points = np.array([img_pairs[1]["kpts1"][ind].pt for ind in points_idx])

        # calculate PnP:
        success, R1, t1 = cv2.solvePnP(objectPoints=world_points.transpose(), 
                            imagePoints=image_points,
                            cameraMatrix=P[:,:3], 
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



    # Xc = RXw + T = [R | T]
    # To find camera position, we have to find Xw such that Xc = 0
    # Xw = inv(R)Xc - inv(R)T = [inv(R) | -inv(R)T]
    # Because R is unitary: inv(R) = R':
    # Xw = R'Xc - R'T = [R' | -R'T]

    logging.info("frame left {} PnP solution: R={}".format(1, R1))
    logging.info("frame left {} PnP solution: T={}".format(1, t1.transpose()))

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
    yaw_pitch_roll.append(cv2.decomposeProjectionMatrix(projMatrix=np.hstack((camRot_left[0], camPos_left[0])), cameraMatrix=K)[6])

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    for idx in range(0, 2):
        # calculating position of right camera
        camPos_right.append(camPos_left[idx] - M2[:, -1].reshape(-1, 1))
        camRot_right.append(camRot_left[idx])
        yaw_pitch_roll.append(cv2.decomposeProjectionMatrix(projMatrix=np.hstack((camRot_left[idx], camPos_left[idx])), cameraMatrix=K)[6])
        logging.info("frame {} LEFT:  cam Pos={}[m]; yaw={}[deg], pitch={}[deg], roll={}[deg]".format(idx, camPos_left[idx].transpose()[0], yaw_pitch_roll[idx][0], yaw_pitch_roll[0][1], yaw_pitch_roll[idx][2]))
        logging.info("frame {} RIGHT: cam Pos={}[m]; yaw={}[deg], pitch={}[deg], roll={}[deg]".format(idx, camPos_right[idx].transpose()[0], yaw_pitch_roll[idx][0], yaw_pitch_roll[idx][1], yaw_pitch_roll[idx][2]))
        plt.scatter(camPos_left[idx][0], camPos_left[idx][2], marker='o', alpha=0.5, facecolors="None", color='blue')
        plt.scatter(camPos_right[idx][0], camPos_right[idx][2], marker='^', alpha=0.5, facecolors="None", color='blue')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    # ax.set_zlabel('Z [m]')

    #
    
    plt.show()
    cv2.waitKey(1)