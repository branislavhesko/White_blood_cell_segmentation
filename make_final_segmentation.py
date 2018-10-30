import numpy as np
import os
import sys
import glob
import shutil
import cv2

what = "test"
path  = "C:/Users/hesko/Desktop/segmentation_result/" + what + "/"
filename = "./" + what + ".csv"

disc_folder = "./ckpt/refuge/7_117/"
cup_folder = "./ckpt/refuge_cup/729_3/"


imgs = glob.glob(disc_folder + "*.jpg")
masks_disc = glob.glob(disc_folder + "*_segmented.bmp")
masks_cup = glob.glob(cup_folder + "*_otsu.bmp")

def get_centers(filename):
    centers = {}
    file = open(filename)

    for line in file.readlines():
        ln = line.split(" ")
        centers[ln[0]] = (int(ln[1]), int(ln[2]))

    return centers

crop_size = 256
centers = get_centers(filename)
print(masks_cup)
for img, disc, cup in zip(imgs, masks_disc, masks_cup):
    img_name = os.path.split(img)[-1][:-4]
    shutil.copy(img, path + img_name + ".jpg")
    disc = cv2.imread(disc, cv2.IMREAD_GRAYSCALE)
    final_mask = np.ones(disc.shape, dtype=np.uint8) * 255
    final_mask[disc > 0] = 128
    cup = cv2.imread(cup, cv2.IMREAD_GRAYSCALE)
    cup_total = np.zeros(disc.shape, dtype=np.uint8)
    x, y = centers[img_name]
    cup_total[x-crop_size//2:x+crop_size//2, y-crop_size//2:y+crop_size//2] = cup
    final_mask[cup_total > 0] = 0
    cv2.imwrite(path + img_name +"_seg.bmp", final_mask)