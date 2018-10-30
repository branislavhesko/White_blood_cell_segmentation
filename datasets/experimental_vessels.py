import os
import glob
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import cv2

num_classes = 2
ignore_label = 0
root = './experimental_fundus_vessels'
#root = "C:/Users/hesko/Desktop/segmentation_result"
#print(root)
'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 0, 255, 0]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask



def make_dataset(mode):
    assert mode in ["training", "test", "validate"]
    items = []

    if mode == "training":
        img_path = os.path.join(root, "train", "original")
        mask_path = os.path.join(root, "train", "masks")
    elif mode == "validate":
        img_path = os.path.join(root, "validate", "imgs")
        mask_path = os.path.join(root, "validate", "masks")
    else:
        img_path = os.path.join(root, "test", "imgs/pngs3")
        mask_path = os.path.join(root, "test", "masks")

    imgs = glob.glob(img_path + "/*.png")
    masks = glob.glob(mask_path + "/*.png")
    data = []
    for img, mask in zip(imgs, masks):
        data.append((img,mask))

    return data

class Retinaimages(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path).convert("L")
        # img, mask = Image.fromarray(cv2.imread(img_path, cv2.IMREAD_COLOR)), cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask)
        #mask_copy = mask.copy()
        #for k, v in self.id_to_trainid.items():
        #    mask_copy[mask == k] = v
        #mask = Image.fromarray(mask_copy.astype(np.uint8))
        mask[mask > 0] = 1
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            # print(len(img_slices))
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)
    



if __name__ == "__main__":
    print("hello")
    print(make_dataset("training"))