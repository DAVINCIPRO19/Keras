from skimage.transform import resize
from skimage.io import imread
from tensorflow import keras
import numpy as np

class PersonDataset(keras.utils.Sequence):

    def __init__ (self, img_sz, num_channels, img_path, mask_path):
        self.img_sz = img_sz
        self.num_channels = num_channels
        self.img_path = img_path
        self.mask_path = mask_path

    def __len__ (self):
        return len(self.img_path)

    def __getitem__ (self, idx):
        img = np.array(resize(imread(self.img_path[idx]), self.img_sz + (self.num_channels, )))
        mask = np.array(resize(imread(self.mask_path[idx]), self.img_sz + (1, )))

        return img, mask