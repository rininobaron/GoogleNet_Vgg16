# -*- coding: utf-8 -*-
# Elaborado por Ricardo Niño de Rivera Barrón 
# Todos los Derechos Reservados
# 05/12/2019

# import the necessary packages
import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model

# load minit data
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# plot 6 images as gray scale
import matplotlib.pyplot as plt

# Con estas instrucciones utilizamos la gpu
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

#Aquí importamos el modelo
import googlenet_he

plt.subplot(321)
plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(322)
plt.imshow(x_train[1],cmap=plt.get_cmap('gray'))
plt.subplot(323)
plt.imshow(x_train[2],cmap=plt.get_cmap('gray'))
plt.subplot(324)
plt.imshow(x_train[3],cmap=plt.get_cmap('gray'))
plt.subplot(325)
plt.imshow(x_train[4],cmap=plt.get_cmap('gray'))
plt.subplot(326)
plt.imshow(x_train[5],cmap=plt.get_cmap('gray'))
# show
plt.show()

# reshape the data to four dimensions, due to the input of model
# reshape to be [samples][width][height][pixels]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')


# normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# parameters
EPOCHS = 1
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 10
norm_size = 28

# start to train model
print('start to train model')

model = googlenet_he.GoogleNet.build(width=norm_size, height=norm_size, depth=1, NB_CLASS=CLASS_NUM)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# Use generators to save memory
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),
                            steps_per_epoch=len(x_train) // BS, validation_data=aug.flow(x_test, y_test),
                            epochs=EPOCHS, verbose=2, use_multiprocessing=False)

# save model
# method one
model.save('googlenet.h5')

# method two
# save model by json and weights
# save json
from keras.models import model_from_json
json_string = model.to_json()
with open(r'googlenet.json', 'w') as file:
    file.write(json_string)

# save weights
model.save_weights('googlenet.h5')

# load model
# method one
# model.load('googlenet.h5')

# model two
# load model by json and weights
# with open(r'googlenet.json', 'r') as file:
#     model_json1 = file.read()
#
# model = model_from_json(json_string)
# model.load_weights('googlenet.h5', by_name=True)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# plot the iteration process
N = EPOCHS
plt.figure()
plt.plot(np.arange(0,N),H.history['loss'],label='loss')
plt.plot(np.arange(0,N),H.history['accuracy'],label='train_acc')
plt.title('Training Loss and Accuracy on mnist-img classifier')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig('FIGURA.png')

# Calculating loss and accuracy
# train
tr_loss, tr_accurary = model.evaluate(x_train, y_train)
# tr_loss = 0.039, tr_accurary = 0.98845
# test
te_loss, te_accurary = model.evaluate(x_test, y_test)
# te_loss = 0.042, te_accurary = 0.9861


