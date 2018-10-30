from scipy.misc import imresize
import numpy as np
import os
import sys
import cv2
import glob

basic = "./hrf"
change_to = "./hrf2"
folders = next(os.walk(basic))[1]
scale = 1. / 2.
print(folders)


for folder in folders:
    subfolders = next(os.walk(os.path.join(basic, folder)))[1]

    if not os.path.exists(os.path.join(change_to, folder)):
        os.mkdir(os.path.join(change_to, folder))
    print(subfolders)

    for subfolder in subfolders:

        if not os.path.exists(os.path.join(change_to, folder, subfolder)):
            os.mkdir(os.path.join(change_to, folder, subfolder))

        imgs = glob.glob(os.path.join(basic, folder, subfolder) + "/*.jpg")
        masks = glob.glob(os.path.join(basic, folder, subfolder) + "/*.tif")

        for i in imgs:
            img = cv2.imread(i, cv2.IMREAD_COLOR)
            new_shape = np.array(img.shape[:-1]) * scale
            cv2.imwrite(i.replace(basic, change_to),
                        imresize(img, (*new_shape.astype(int), 3)))
        for m in masks:
            img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            new_shape = np.array(img.shape) * scale
            print(new_shape)
            cv2.imwrite(m.replace(basic, change_to),
                        imresize(img, new_shape.astype(int)))

        print(imgs)
        print(masks)
