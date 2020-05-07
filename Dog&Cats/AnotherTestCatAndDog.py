from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras import backend as k
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import os

# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip',origin=_URL,extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip),'cats_and_dogs_filtred')
# train_dir = os.path.join(PATH,'train')
# validation_dir = os.path.join(PATH,'validation')

train_dir = 'C:\\Users\\saber\\.keras\\datasets\\cats_and_dogs_filtered\\train'
validation_dir = 'C:\\Users\\saber\\.keras\\datasets\\cats_and_dogs_filtered\\validation'

train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')

validation_cats_dir = os.path.join(validation_dir,'cats')
validation_dogs_dir = os.path.join(validation_dir,'dogs')

nbr_cats_train = len(os.listdir(train_cats_dir))
nbr_dogs_train = len(os.listdir(train_dogs_dir))

nbr_cats_val = len(os.listdir(validation_cats_dir))
nbr_dogs_val = len(os.listdir(validation_dogs_dir))

print('nbr cats train is: ',nbr_cats_train)
print('nbr dogs train is: ',nbr_dogs_train)

print('nbr cats valid is: ',nbr_cats_val)
print('nbr dogs valid is: ',nbr_dogs_val)

total_train = 1000
total_val = 100

batch_size = 20
epochs = 10
img_height = 150
img_width = 150
#data Preparation

#cheking if the image in the RGB format
if k.image_data_format()=='channels_first':
    #know how much data you working with
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.\
    flow_from_directory(batch_size=batch_size,
                        directory=train_dir,
                        shuffle=True,
                        target_size=(img_height, img_width),
                        class_mode='binary')

val_data_gen = validation_image_generator.\
    flow_from_directory(batch_size=batch_size,
                        directory=validation_dir,
                        target_size=(img_height, img_width),
                        class_mode='binary')

#create the model

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
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])
#View all the layers of the network using the model's summary method
model.summary()


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps= total_val // batch_size
)

model.save_weights('results.h5')

# model = load_model('results.h5')

img_pred = image.load_img('C:\\Users\\saber\\.keras\\datasets\\cats_and_dogs_filtered\\validation\\dogs\\dog.2013.jpg',target_size=(150,150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred,axis = 0)
#############
result = model.predict(img_pred)
print("results= ",result)


predection=''
if result[0][0] == 1:
    predection = "dog"
else:
    predection = "cat"

print(predection)













