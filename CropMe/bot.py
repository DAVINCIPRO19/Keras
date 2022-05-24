import telebot
import config
import cv2
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import tensorflow as tf
from tensorflow import keras


model = keras.models.load_model('tmp/model128')


def get_crop(fdir='photo/BQACAgIAAxkBAANjYoo5dygtqg_bnxWLr9Dl5VN3v10AAgkZAALFzQABSEVtd7k8T634JAQ.jpg'):
    x_test = np.zeros((1, 128, 128, 3), dtype=np.float64)
    origin = np.array(imread(fdir))
    photo = resize(origin, (128, 128, ))
    x_test[0] = photo
    mask = model.predict(x_test)[0]
    mask = tf.image.resize(
        mask,
        (origin.shape[0], origin.shape[1]),
        antialias=False,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    _, mask = cv2.threshold(np.array(mask), 0.1, 1, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(np.array(mask, dtype=np.uint8), (3, 3), cv2.BORDER_DEFAULT)
    cropped = cv2.bitwise_and(origin, origin, mask=mask)
    return cropped

bot = telebot.TeleBot(config.TOKEN)

@bot.message_handler(content_types=['document'])
def crop(message):
    print(message.document)
    fid = message.document.file_id
    finfo = bot.get_file(fid)
    fdir = 'photo/' + fid + '.jpg'
    photo = bot.download_file(finfo.file_path)
    with open(fdir, 'wb') as file:
        file.write(photo)
    cropped = get_crop(fdir)
    imsave('crop/cropped_{}.png'.format(fid), cropped)
    send = open('crop/cropped_{}.png'.format(fid), 'rb')
    bot.send_photo(message.chat.id, send)
    print('sent!')

bot.polling(none_stop=True)    


