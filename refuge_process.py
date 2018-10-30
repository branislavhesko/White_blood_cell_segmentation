import sys
import os
import glob
import shutil
import random
from PIL import Image
from PIL.ImageOps import grayscale
import numpy as np
from scipy.misc import imresize
import cv2

folder = "./Refuge"
train_test_ratio = (0.85, 0.05, 0.1)

previous_images = glob.glob(os.path.join(folder, "train", "img") + "/*")
previous_masks = glob.glob(os.path.join(folder, "train", "masks") + "/*")
print(previous_images)

previous_img = glob.glob(os.path.join(folder, "test", "img") + "/*")
previous_mask = glob.glob(os.path.join(folder, "test", "masks") + "/*")

previous_img_v = glob.glob(os.path.join(folder, "validate", "img") + "/*")
previous_mask_v = glob.glob(os.path.join(folder, "validate", "masks") + "/*")

for img in previous_images + previous_masks + previous_img + previous_mask + previous_img_v + previous_mask_v:
    os.remove(img)

all_imgs = glob.glob(os.path.join(folder, "Training400", "Glaucoma") + "/*") + \
    glob.glob(os.path.join(folder, "Training400", "Non-Glaucoma") + "/*")

all_masks = glob.glob(os.path.join(folder, "Training400", "Annotation", "Disc_Cup_Masks", "Glaucoma") + "/*") + \
    glob.glob(os.path.join(folder, "Training400", "Annotation",
                           "Disc_Cup_Masks", "Non-Glaucoma") + "/*")


k = list(range(400))
random.shuffle(k)

ratio = 2

# Prepare training dataset
for i in k[:int(len(k) * train_test_ratio[0])]:
    filename = os.path.split(all_imgs[i])[-1]
    target_path = os.path.join(folder, "train", "img") + "/" + filename
    print(target_path)

    mask = grayscale(Image.open(all_masks[i]))
    mask = np.array(mask)
    mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY_INV)[1]
    mask = imresize(mask, (mask.shape[0] // 2, mask.shape[1] // 2 ))

    filename = os.path.split(all_masks[i])[-1]

    target_mask = os.path.join(
        folder, "train", "masks" + "/") + filename[:-4] + ".png"
    cv2.imwrite(target_mask, mask)
    # shutil.copyfile(all_imgs[i], target_path)
    img = Image.open(all_imgs[i])
    img = np.array(img)
    img = imresize(img, (img.shape[0] // 2, img.shape[1] // 2, 3))
    cv2.imwrite(target_path, img[ :, :, -1::-1])

# Prepare test dataset
for i in k[int(len(k) * train_test_ratio[0]):int(len(k) * (train_test_ratio[0] + train_test_ratio[1]))]:
    filename = os.path.split(all_imgs[i])[-1]
    target_path = os.path.join(folder, "validate", "img") + "/" + filename
    print(target_path)

    mask = grayscale(Image.open(all_masks[i]))
    mask = np.array(mask)
    mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY_INV)[1]
    mask = imresize(mask, (mask.shape[0] // 2, mask.shape[1] // 2))
    filename = os.path.split(all_masks[i])[-1]

    target_mask = os.path.join(
        folder, "validate", "masks" + "/") + filename[:-4] + ".png"
    cv2.imwrite(target_mask, mask)
    #shutil.copyfile(all_imgs[i], target_path)
    img = Image.open(all_imgs[i])
    img = np.array(img)
    img = imresize(img, (img.shape[0] // 2, img.shape[1] // 2, 3))
    cv2.imwrite(target_path, img[ :, :, -1::-1])


for i in k[int(len(k) * (1 - train_test_ratio[2])):]:
    filename = os.path.split(all_imgs[i])[-1]
    target_path = os.path.join(folder, "test", "img") + "/" + filename
    print(target_path)

    mask = grayscale(Image.open(all_masks[i]))
    mask = np.array(mask)
    mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY_INV)[1]
    mask = imresize(mask, (mask.shape[0] // 2, mask.shape[1] // 2))

    filename = os.path.split(all_masks[i])[-1]

    target_mask = os.path.join(
        folder, "test", "masks" + "/") + filename[:-4] + ".png"
    cv2.imwrite(target_mask, mask)
    #shutil.copyfile(all_imgs[i], target_path)
    img = Image.open(all_imgs[i])
    img = np.array(img)
    img = imresize(img, (img.shape[0] // 2, img.shape[1] // 2, 3))
    cv2.imwrite(target_path, img[ :, :, -1::-1])
