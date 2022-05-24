from tensorflow import keras
from tensorflow.keras import layers

class UNET:

    def __init__(self, img_sz, num_channels):

        self.inputs = layers.Input(img_sz + (num_channels, ))
        self.lambda_ = layers.Lambda(lambda x: x / 255.)

        self.conv1_1 = layers.Conv2D(112, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.conv1_2 = layers.Conv2D(112, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.maxpool_1 = layers.MaxPool2D((2, 2))

        self.conv2_1 = layers.Conv2D(224, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.conv2_2 = layers.Conv2D(224, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.maxpool_2 = layers.MaxPool2D((2, 2))

        self.conv3_1 = layers.Conv2D(448, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.conv3_2 = layers.Conv2D(448, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.maxpool_3 = layers.MaxPool2D((2, 2))

        self.conv4_1 = layers.Conv2D(448, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.conv4_2 = layers.Conv2D(448, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.upconv4 = layers.Conv2DTranspose(448, (2, 2), strides=(2, 2), padding='same')

        self.dropout_5 = layers.Dropout(0.1)
        self.conv5_1 = layers.Conv2D(224, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.conv5_2 = layers.Conv2D(224, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.upconv5 = layers.Conv2DTranspose(224, (2, 2), strides=(2, 2), padding='same')

        self.dropout_6 = layers.Dropout(0.1)
        self.conv6_1 = layers.Conv2D(112, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.conv6_2 = layers.Conv2D(112, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.upconv6 = layers.Conv2DTranspose(112, (2, 2), strides=(2, 2), padding='same')

        self.dropout_7 = layers.Dropout(0.1)
        self.conv7_1 = layers.Conv2D(112, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.conv7_2 = layers.Conv2D(112, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def build(self):
        inputs = self.inputs
        inputs = self.lambda_(inputs)

        l1 = self.conv1_1(inputs)
        l1 = self.conv1_2(l1)

        l2 = self.conv2_1(self.maxpool_1(l1))
        l2 = self.conv2_2(l2)

        l3 = self.conv3_1(self.maxpool_2(l2))
        l3 = self.conv3_2(l3)

        l4 = self.conv4_1(self.maxpool_3(l3))
        l4 = self.conv4_2(l4)

        l5 = layers.concatenate([l3, self.upconv4(l4)])
        l5 = self.dropout_5(l5)
        l5 = self.conv5_1(l5)
        l5 = self.conv5_2(l5)

        l6 = layers.concatenate([l2, self.upconv5(l5)])
        l5 = self.dropout_6(l6)
        l6 = self.conv6_1(l6)
        l6 = self.conv6_2(l6)

        l7 = layers.concatenate([l1, self.upconv6(l6)])
        l7 = self.dropout_7(l7)
        l7 = self.conv7_1(l7)
        l7 = self.conv7_2(l7)

        outputs = self.outputs(l7)

        return keras.Model(inputs=[inputs], outputs=[outputs])