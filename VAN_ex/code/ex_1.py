import cv2
import numpy as numpy

DATA_PATH = r'./VAN_ex/data/dataset05/sequences/05/'
NUM_KEYPTS = 1000

def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + "image_0/" + img_name, 0)
    img2 = cv2.imread(DATA_PATH + "image_1/" + img_name, 0)
    return img1, img2




if __name__ == "__main__":
    idx = 0
    img1, img2 = read_images(idx)
    cv2.imshow("img1", img1)
    cv2.waitKey(0)