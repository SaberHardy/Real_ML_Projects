from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as k
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

#dimention of our images
img_width = img_height = 150

train_data_dir = 'F:\\ML\\DataSets\\catDogDataSet\\catdog\\train'
validation_data_dir = 'F:\\ML\\DataSets\\catDogDataSet\\catdog\\validation'

nb_train_samples = 1000
nb_validation_samples = 100
epochs = 50
#number of images you will give in one time
batch_size = 20
#cheking if the image in the RGB format
if k.image_data_format()=='channels_first':
    #know how much data you working with
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

train_data_gen = ImageDataGenerator(
    rescale=1. /255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#rescale Data
test_data_gen = ImageDataGenerator(rescale=1./255)

traing_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_data_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary')

#make Neural network
model = Sequential()
#extract 32 features from the image given 3 cross 3
model.add(Conv2D(32,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.summary()

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#64 features dataset
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#flatten it's for 2D img to 1D img
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#one output
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrixs=['accuracy'])

#this is the augmentation configuration we will use for Training
#excute the Neural Networks
model.fit_generator(
    train_data_gen,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

model.save_weights('results.h5')

img_pred = image.load_img('F:\\ML\DataSets\\catDogDataSet\\validation\\7.jpg',target_size=(150,150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred,axis = 0)
#############
result = model.predict(img_pred)
print(result)
if result[0][0] ==1:
    predection = "Dog"
else:
    prediction = "Cat"
print(predection)















