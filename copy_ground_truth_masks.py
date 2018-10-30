import shutil
import numpy as np
import glob
import sys
import os
import cv2


path_to_masks = "./Refuge/Training400/Annotation/Disc_Cup_Masks/"

gt_masks = glob.glob(path_to_masks + "Glaucoma/*.bmp") + glob.glob(path_to_masks + "Non-Glaucoma/*.bmp")


destination = "./Refuge/test/masks/"
img_names = glob.glob("./Refuge/test/img/*.jpg")


for img in img_names:
