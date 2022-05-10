!pip install --upgrade --no-cache-dir gdown

from IPython.display import clear_output 
clear_output()

# Step 1 : Git clone Feature map

'''
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo/5_TF2_UCF101_video_classification.git
! git pull origin master
# ! git pull origin main
'''

# Mini-COCO dataset download from Auther's Github repository
import gdown
google_path = 'https://drive.google.com/uc?id='
file_id = '18I06ymkUqKwEon4Dsb8GqkJORwPZZxLd'
output_name = 'Cifar_10.zip'
gdown.download(google_path+file_id,output_name,quiet=False)
# https://drive.google.com/file/d/18I06ymkUqKwEon4Dsb8GqkJORwPZZxLd/view?usp=sharing

% rm -rf sample_data
!unzip /content/Cifar_10.zip -d /content/data
clear_output()
! rm /content/Cifar_10.zip

import os
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras import Model, Sequential

batch_size = 128
n_epoch = 3

# path joining version for other paths


Datasets = "cifar_10_224_pixels"



n_classes = 10
img_size = 224       # 224

dst_dir_train = '/content/data/train/'
dst_dir_test  = '/content/data/test/'
    

num_train = sum([len(files) for r, d, files in os.walk(dst_dir_train)])
num_test  = sum([len(files) for r, d, files in os.walk(dst_dir_test)])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_ds = train_datagen.flow_from_directory(dst_dir_train,
                                                 target_size = (img_size, img_size),
                                                 batch_size = batch_size,
                                                 class_mode = 'sparse')
                                                 # class_mode = 'categorical')

test_ds = test_datagen.flow_from_directory(dst_dir_test,
                                            target_size = (img_size, img_size),
                                            batch_size = batch_size,
                                            class_mode = 'sparse')
                                            # class_mode = 'categorical')

steps_per_epoch  = int(num_train/batch_size)
validation_steps = int(num_test/batch_size)


#Define network
IMG_SIZE = 224                      # VGG19
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = 10                    # cifar10

# 사전 훈련된 모델 VGG19 에서 기본 모델을 생성합니다.
base_model = tf.keras.applications.ResNet152V2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.summary()

# The last 15 layers fine tune
for layer in base_model.layers[:-15]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
output  = Dense(units=num_classes, activation='softmax')(x)
model = Model(base_model.input, output)


model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_name = 'cifar10_ResNet152V2'


"""
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('model_best_weights.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
model.fit_generator(X_train, Y_train, validation_data=(X_val, Y_val), 
      callbacks = [early_stop,checkpoint])
"""

import time

start_time = time.time()

model.fit_generator(train_ds,steps_per_epoch = steps_per_epoch, epochs = n_epoch, 
                    validation_data = test_ds, validation_steps = validation_steps)

# model.evaluate_generator(test_set, validation_steps)

finish_time = time.time()

print(int(finish_time - start_time), "Sec")

