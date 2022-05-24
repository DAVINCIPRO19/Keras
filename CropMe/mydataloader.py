from mydataset import PersonDataset
import numpy as np

class PersonDataloader:

    def __init__(self, batch_sz, img_sz, num_channels, img_path, mask_path):
        self.batch_sz = batch_sz
        self.img_sz = img_sz
        self.num_channels = num_channels
        self.img_path = img_path
        self.mask_path = mask_path

    def __getitem__(self, idx):
        i = idx * self.batch_sz
        if i > len(self.img_path):
            raise StopIteration
        print(' ' + str(i) + " ", end='')

        batched_img_path = self.img_path[i: i + self.batch_sz]
        batched_mask_path = self.mask_path[i: i + self.batch_sz]

        x = np.zeros(((self.batch_sz, ) + self.img_sz + (self.num_channels, )), dtype=np.float64)
        y = np.zeros(((self.batch_sz, ) + self.img_sz + (1, )))

        temp_dataset = PersonDataset(self.img_sz, self.num_channels, batched_img_path, batched_mask_path)

        for idx, (img, mask) in enumerate(temp_dataset):
            x[idx] = img
            y[idx] = mask
        
        return x, y