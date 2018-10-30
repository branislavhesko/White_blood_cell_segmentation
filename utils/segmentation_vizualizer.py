import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio

# All images have to be uint8

def vizualize_segmentation(img1, img2):
    result_img = np.zeros((*img1.shape, 3))
    print(result_img.shape)
    intersetcion = cv2.bitwise_and(img1, img2)
    img1[intersetcion > 0] = 0
    img2[intersetcion > 0] = 0
    result_img[intersetcion > 0, 1] = 255
    result_img[img1 > 0, 0] = 255
    result_img[img2 > 0, 2] = 255
    return result_img.astype(np.uint8)

def store_predictions(filename, matrix):
    sio.savemat(filename, matrix)

def store_vizualization(filename, img):
    plt.figure(0, figsize=(16,9), dpi=100)
    plt.imshow(img)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def show_segmentation_into_original_image(img, segmented_img):
    img[segmented_img > 0, 0] = 0
    img[segmented_img > 0, 2] = 0
    return img

if __name__ == "__main__":
    img1 = cv2.imread("g0001.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("g0001.png", cv2.IMREAD_GRAYSCALE)


    store_vizualization("okno.png", show_segmentation_into_original_image(img1, img2)[:,:,::-1])
