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
from matplotlib import pyplot as plt

all_imgs = glob.glob(os.path.join(folder, "Training400", "Glaucoma") + "/*") + \
    glob.glob(os.path.join(folder, "Training400", "Non-Glaucoma") + "/*")

all_masks = glob.glob(os.path.join(folder, "Training400", "Annotation", "Disc_Cup_Masks", "Glaucoma") + "/*") + \
    glob.glob(os.path.join(folder, "Training400", "Annotation",
                           "Disc_Cup_Masks", "Non-Glaucoma") + "/*")


cup_to_disc_ratio_g = []
cup_to_disc_ratio_n = []

numbers = np.zeros(2)

for msk in all_masks:
	img = np.array(grayscale(Image.open(msk)))

	cup = np.sum(img < 10)
	disc = np.sum(img < 150)

	if os.path.split(msk)[-1][0] == "g":
		print("Glaucomatic {}".format(cup/disc))
		cup_to_disc_ratio_g.append(cup / disc)
		numbers[1] += 1
	else:
		cup_to_disc_ratio_n.append(cup / disc)
		print("Non-glaucomatic {}".format(cup/disc))
		numbers[0] += 1

print("Glaucomatic c/d ratio {}".format(np.mean(cup_to_disc_ratio_g)/ numbers[1]))
print("Non-Glaucomatic c/d ratio {}".format(np.mean(cup_to_disc_ratio_n)/ numbers[0]))
plt.boxplot((cup_to_disc_ratio_g, cup_to_disc_ratio_n), labels = ("Glaucomatic", "Non-Glaucomatic"))
plt.show()