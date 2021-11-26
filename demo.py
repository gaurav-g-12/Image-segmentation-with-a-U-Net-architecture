from PIL import ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
path  = r'E:\Project\MiniProject_AV\bvb_images\1_json\label.png'
image = ImageOps.autocontrast(tf.keras.preprocessing.image.load_img(path))
plt.imshow(image)
plt.show()
image = numpy.array(image)
print(image)

























