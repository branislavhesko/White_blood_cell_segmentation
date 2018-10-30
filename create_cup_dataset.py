import glob
import os
import sys
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import cv2
from PIL import Image
from PIL.ImageOps import grayscale


path = "C:\\Users\\hesko\\Desktop\\segmentation_training_data\\7_10002\\"
centers = open("train.csv", "w")
masks = glob.glob(path + "*.bmp")
masks = [mask for mask in masks if mask.find("gt") < 0 and mask.find("segmented") > 0 and mask.find("ground") < 0]
images = glob.glob(path + "*.jpg")
print(masks)
masks_cup = glob.glob(path + "*_ground_truth.bmp")

crop_size = 512

destination = "./refuge_cup_dataset/train/"
index = 0
for mask, img, cup in zip(masks, images, masks_cup):
    msk = np.array(grayscale(Image.open(mask)))
    msk = cv2.resize(msk, (1024, 1024))
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    msk_cup =  cv2.threshold(np.array(grayscale(Image.open(cup))), 50, 255, cv2.THRESH_BINARY)[1]
    msk = cv2.threshold(msk, 100, 255, cv2.THRESH_BINARY)[1]
    x, y = center_of_mass(msk)
    img_name = os.path.split(img)[-1][:-4]
    x = int(x)
    y = int(y)

    if y < crop_size // 2:
        y = crop_size // 2
    centers.write(" ".join([img_name, str(x), str(y)])+ "\n")

    cv2.imwrite(destination + "masks/" + img_name + ".png", msk_cup[x-crop_size//2:x+crop_size//2, y-crop_size//2:y+crop_size//2])
    cv2.imwrite(destination + "img/" + img_name + ".jpg", image[x-crop_size // 2:x + crop_size // 2, y-crop_size//2:y+crop_size//2, :])
    print(img_name)
    index += 1

centers.close()