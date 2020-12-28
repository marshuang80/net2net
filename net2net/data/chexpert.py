import os
import json
import albumentations
import numpy as np
import cv2 
import pandas as pd
import PIL

from torch.utils.data import Dataset
from tqdm             import tqdm
from torchvision      import transforms
from typing           import Union
from pathlib          import Path
from PIL              import Image


CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]
VIEW_COL = "Frontal/Lateral"
PATH_COL = "Path"
SPLIT_COL = "DataSplit"
REPORT_COL = "Report Impression"
PROJECT_DATA_DIR = Path("/data4/embeddingx")    # need to modify
CHEXPERT_DIR = PROJECT_DATA_DIR / "CheXpert"
CHEXPERT_DATA_DIR = CHEXPERT_DIR / "CheXpert-v1.0"
CHEXPERT_RAD_CSV = CHEXPERT_DATA_DIR / "master.csv"

# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CheXpertBase(Dataset):
    """Base chexpert dataset"""

    def __init__(
            self,
            data_transform: transforms.Compose,
            img_type: str = "Frontal", 
            resize_shape: float = 256, 
            split: str = "train"
            ):
        """Constructor for dataset class
        Args: 
            csv_path (str): path to csv file containing paths to jpgs
            data_transfrom (transforms.Compose): data transformations
            img_type (str): type of xray images to use ["All", "Frontal", "Lateral"]
            uncertain (str): how to handel uncertain cases ["ignore", "zero", "one"]
            split (str): datasplit to parse in a dataloader
        """
        # read in csv file
        self.df = pd.read_csv(CHEXPERT_RAD_CSV)

        # filter dataframe by split
        self.df = self.df[self.df[SPLIT_COL] == split]

        # filter image type 
        if img_type != "All":
            self.df = self.df[self.df[VIEW_COL] == img_type]

        # get column names of the target labels
        self.label_cols = self.df.columns[-14:]

        self.data_transform = data_transform
        self.resize_shape = resize_shape

    def __len__(self):
        '''Returns the size of the dataset'''
        return len(self.df)

    def __getitem__(self, idx):
        img = self._get_image(idx)
        report = self.df.iloc[idx][REPORT_COL]

        example = {"image": img,
                   "caption": [report]}
        return example


    def _get_image(self, idx):
        """Read in image and apply transformations based on index"""

        path = os.path.join(CHEXPERT_DATA_DIR, '/'.join(self.df.iloc[idx][PATH_COL].split('/')[1:]))

        x = cv2.imread(str(path), 0)

        # tranform images 
        x = self._resize_img(x, self.resize_shape)
        x = Image.fromarray(x).convert('RGB')
        if self.data_transform is not None:
            x = self.data_transform(x)

        # channel last 
        x = x.permute(1,2,0)
        
        return x

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        #Resizing
        if max_ind == 0:
            #image is heigher
            wpercent = (scale / float(size[0]))
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            #image is wider
            hpercent = (scale / float(size[1]))
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(img, desireable_size[::-1], interpolation = cv2.INTER_AREA) #this flips the desireable_size vector

        #Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size/2))
            right = int(np.ceil(pad_size/2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size/2))
            bottom = int(np.ceil(pad_size/2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(resized_img,[(top, bottom), (left, right)], 'constant', constant_values=0)

        return resized_img

class CheXpertImagesAndCaptionsTrain(CheXpertBase):
    """returns a pair of (image, caption)"""

    def __init__(self, img_type='Frontal', resize_shape=256, crop_shape=224, rotation_range=20):

        # TODO: removed cropping 
        data_transform = [
            #transforms.RandomCrop((crop_shape, crop_shape)),
            #transforms.RandomRotation(rotation_range, resample=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
        data_transform = transforms.Compose(data_transform)

        super().__init__(data_transform=data_transform,
                         img_type=img_type,
                         resize_shape=resize_shape,
                         split='train'
                         )

    def get_split(self):
        return "train"

class CheXpertImagesAndCaptionsValidation(CheXpertBase):
    """returns a pair of (image, caption)"""

    def __init__(self, img_type='Frontal', resize_shape=256, crop_shape=224):

        data_transform = [
            #transforms.CenterCrop((crop_shape, crop_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
        data_transform = transforms.Compose(data_transform)

        super().__init__(data_transform=data_transform,
                         img_type=img_type,
                         resize_shape=resize_shape,
                         split='valid'
                         )

    def get_split(self):
        return "validation"
