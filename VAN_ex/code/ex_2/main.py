import os
from utils import *
import matplotlib.pyplot as plt

DATA_PATH = r'../../data/dataset05/sequences/05/'

if __name__ == "__main__":
    """ [2.1] match image pair #1 """
    print("\n")
    logging.info("Begin EX 2")
    img_pair=[]
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
        img_pair.append(create_img_pair_from_img_dicts(left_imgs[idx], right_imgs[idx]))
        # match key points
        img_pair[idx] = match_rectified_images(img_pair[idx], feature_matcher="bf", cumsumThres = 0.88, plot=False)
        draw_img_pair_kpts(img_pair[idx], title="matched pairs:[left {}, right {}]")

        # get point cloud
        point_cloud.append(generate_point_cloud(img_pair[idx], P, Q, plot=True))
        logging.info("calculated 3D point cloud for image pair {}".format(idx))

    
    # match points in two left images
    print("\n")
    logging.info("===== Answer [2.2] =====")
    left_frames_seq = []
    for idx in range(0, 1):
        left_frames_seq.append(create_img_pair_from_img_dicts(left_imgs[idx], left_imgs[idx + 1])) 
        left_frames_seq[idx] = match_images(left_frames_seq[idx], matcher="bf", cumsumThres=0.88, plot=False)
        draw_img_pair_kpts(left_frames_seq[idx], title="matched seq:[left {}, left {}]")
        logging.info("matched features in frame seq: left[{}], left[{}]".format(idx, idx+1))
    
    print("\n")
    logging.info("===== Answer [2.3] =====")
    # finding kpts that were matches between subsequant left image frames, and between left and right image frames
    # thest are matches that are found in: inliers[left0, right0]; inliers[left0, left1]; inliers[left1, right1]
    for frame_idx in range(0, 1):
        frame_seq, img_pair = get_consistent_matches_between_frames(left_frames_seq[frame_idx], img_pair)


    
    plt.show()
    cv2.waitKey(0)
