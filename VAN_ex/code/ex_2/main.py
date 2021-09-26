import os
from utils import *
import matplotlib.pyplot as plt
import random

DATA_PATH = r'../../data/dataset05/sequences/05/'

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
                "kpts1": img1_dict["kpts"],
                "desc1": img1_dict["desc"],
                "kpts2": img2_dict["kpts"],
                "desc2": img2_dict["desc"],
                "left_img": img1_dict["img"],
                "right_img": img2_dict["img"],
                "method": "features",
                "inliers": macth inliers,
                "outliers":  match outliers
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
        img_pairs[idx] = match_rectified_images(img_pairs[idx], feature_matcher="bf", cumsumThres = 0.88, plot=False)
        draw_img_pair_kpts(img_pairs[idx], title="matched pairs:[left {}, right {}]")

        # get point cloud
        point_cloud.append(generate_point_cloud(img_pairs[idx], P, Q, plot=True))
        logging.info("calculated 3D point cloud for image pair {}".format(idx))

    
    # match points in two left images
    print("\n")
    logging.info("===== Answer [2.2] =====")
    left_cam_pairs = []
    for idx in range(0, 1):
        left_cam_pairs.append(create_img_pair_from_img_dicts(left_imgs[idx], left_imgs[idx + 1])) 
        left_cam_pairs[idx] = match_images(left_cam_pairs[idx], matcher="bf", cumsumThres=0.88, plot=False)
        draw_img_pair_kpts(left_cam_pairs[idx], title="matched seq:[left {}, left {}]")
        logging.info("matched features in frame seq: left[{}], left[{}]".format(idx, idx+1))
    
    print("\n")
    logging.info("===== Answer [2.3] =====")
    # finding kpts that were matches between subsequant left image frames, and between left and right image frames
    # thest are matches that are found in: inliers[left0, right0]; inliers[left0, left1]; inliers[left1, right1]
    for frame_idx in range(0, 1):
        left_cam_pairs[frame_idx], img_pairs = get_consistent_matches_between_frames(left_cam_pairs[frame_idx], img_pairs)


    # randomly select 4 key-points that were matched on all four images (2 image pairs from frame0 and frame 1)
    idx = 0
    # building point cloud only from key points that are matched on all four images (2 image pairs from frame0 and frame1)
    points_idx = random.sample(range(len(left_cam_pairs[idx]["inliers"])), 4)
    world_points = generate_point_cloud(img_pairs[idx], P, Q, inliers_idx=points_idx, plot=True)
    generate_point_cloud(img_pairs[idx+1], P, Q, inliers_idx=points_idx, plot=True)

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
    R1, jacobian = cv2.Rodrigues(R1)
    # Xc = RXw + T = [R | T]
    # To find camera position, we have to find Xw such that Xc = 0
    # Xw = inv(R)Xc - inv(R)T = [inv(R) | -inv(R)T]
    # Because R is unitary: inv(R) = R':
    # Xw = R'Xc - R'T = [R' | -R'T]

    logging.info("frame left {} PnP solution: R={}".format(1, R1))
    logging.info("frame left {} PnP solution: T={}".format(1, t1.transpose()))

    camPos = -R1.transpose() @ t1
    camRot = R1.transpose()

    # get camera rotation:
    yaw_pitch_roll_radians = cv2.decomposeProjectionMatrix(projMatrix=R1.transpose, cameraMatrix=K)

    logging.info("frame {} cam Pos: {}".format(1, camPos))

    plt.show()
    cv2.waitKey(0)