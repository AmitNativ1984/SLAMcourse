from utils import *
DATA_PATH = r'../../data/dataset05/sequences/05/'
if __name__ == "__main__":
    idx = 1
    img1, img2 = read_images(idx, DATA_PATH)

    kpts1, desc1 = detect_keyPts(img1)
    kpts2, desc2 = detect_keyPts(img2)

    draw_kpts(img1, img2, kpts1, kpts2, title="kpts[left/right][{}]".format(idx))
