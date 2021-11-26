import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageOps
from tensorflow.keras import layers
import random

image_path = r'E:\Project\Image_segmentation\images'
mask_path = r'E:\Project\Image_segmentation\annotations\trimaps'
image_size = (160,160)
no_of_class = 3
batch_size = 32

image_ind_path = sorted([os.path.join(image_path, fname) for fname in os.listdir(image_path) if fname.endswith('.jpg')])
mask_ind_path = sorted([os.path.join(mask_path, fname) for fname in os.listdir(mask_path) if fname.endswith('.png') and not fname.startswith('.')])


print(len(image_ind_path))
print(len(mask_ind_path))
for i in range(9):
    print(image_ind_path[i]+' | '+mask_ind_path[i])

image = tf.keras.preprocessing.image.load_img(image_ind_path[2])
plt.imshow(image)
plt.show()

img = PIL.ImageOps.autocontrast(tf.keras.preprocessing.image.load_img(mask_ind_path[2]))
plt.imshow(img)
plt.show()

class dataset_creater(tf.keras.utils.Sequence):
    def __init__(self,batch_size,image_size,image_ind_path,mask_ind_path):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_ind_path = image_ind_path
        self.mask_ind_path = mask_ind_path

    def __len__(self):
        return len(self.image_ind_path)//self.batch_size

    def __getitem__(self, item):
        batch_image_path = self.image_ind_path[item*self.batch_size:item*self.batch_size+self.batch_size]
        batch_mask_path = self.mask_ind_path[item*self.batch_size:item*self.batch_size+self.batch_size]
        x = np.zeros((self.batch_size,)+self.image_size+(3,), dtype='float32')
        for j, path in enumerate(batch_image_path):
            image = tf.keras.preprocessing.image.load_img(path, target_size=self.image_size)
            x[j] = image

        y  = np.zeros((self.batch_size,)+self.image_size+(1,), dtype='uint8')
        for j, path in enumerate(batch_mask_path):
            mask = tf.keras.preprocessing.image.load_img(path, target_size=self.image_size)
            mask = tf.image.rgb_to_grayscale(mask)
            y[j] = mask
            y[j] -= 1

        return x,y


def get_model(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x


    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)


        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    return model

tf.keras.backend.clear_session()

model = get_model(image_size, num_classes=no_of_class)
model.summary()


val_samples = 1000
random.Random(1337).shuffle(image_ind_path)
random.Random(1337).shuffle(mask_ind_path)
train_image_paths = image_ind_path[:-val_samples]
train_mask_paths = mask_ind_path[:-val_samples]
val_image_paths = image_ind_path[-val_samples:]
val_mask_paths = mask_ind_path[-val_samples:]

train_gen = dataset_creater(batch_size, image_size, train_image_paths, train_mask_paths)
val_gen = dataset_creater(batch_size, image_size, val_image_paths, val_mask_paths)


model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("oxford_segmentation.h5")
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)



















