from subprocess import call
from mydataloader import PersonDataloader
from mydataset import PersonDataset
from tensorflow.keras import callbacks
from tensorflow import keras
import tensorflow as tf
from model import UNET
import numpy as np
import random
import os

img_dir = '../SemanticSegmentation/data/images/'
mask_dir = '../SemanticSegmentation/data/masks/'
img_sz = (128, 128, )
num_channels = 3
batch_sz = 1024
epochs = 10

img_path = sorted (
    [
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.endswith('.png')
    ]
)

mask_path = sorted (
    [
        os.path.join(mask_dir, fname)
        for fname in os.listdir(mask_dir)
        if fname.endswith('.png')
    ]
)

# val_samples = 500
random.Random(337).shuffle(img_path)
random.Random(337).shuffle(mask_path)
# train_img_path = img_path[:-val_samples]
# train_mask_path = mask_path[:-val_samples]
# val_img_path = img_path[-val_samples:]
# val_mask_path = img_path[-val_samples:]

# val_ds = PersonDataloader(batch_sz, img_sz, num_channels, val_img_path, val_mask_path)

# train = PersonDataset(batch_sz, img_sz, num_channels, train_img_path, train_mask_path)
# val = PersonDataset(batch_sz, img_sz, num_channels, val_img_path, val_mask_path)

keras.backend.clear_session()


model = UNET(img_sz, num_channels).build()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

checkpoint_filepath = '/tmp/checkpoint'
checkpoint = callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weigths_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                save_best_only=True)

callbacks = [callbacks.EarlyStopping(patience=3, monitor='val_loss')]

train_ds = PersonDataloader(batch_sz, img_sz, num_channels, img_path, mask_path)

for idx, (x, y) in enumerate(train_ds):
    print(idx, x.shape, y.shape)

    model.fit(x, y, epochs=epochs, batch_size=16, callbacks=callbacks)

# model.save('tmp/model128x128')

# history = model.fit(train, epochs=epochs, validation_data=val, callbacks=callbacks)

