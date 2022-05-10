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


import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
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


# returns batch_size random samples from either training set or validation set
# resizes each image to (224, 244, 3), the native input size for VGG19
#Define network
IMG_SIZE = 224                      # VGG19
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = 10                    # cifar10

# 3-Layers Convolution neural network with one hidden layer
class CNN_Model(Model):
    def __init__(self):
        super(CNN_Model, self).__init__()
        
        # Convolution 1
        self.conv1 = Conv2D(64, 3, activation='relu', padding='SAME')
        
        # Max pool 1
        self.maxpool2d1 = MaxPool2D(padding='SAME')
     
        # Convolution 2
        self.conv2 = Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu, padding='SAME')
        
        # Max pool 2
        self.maxpool2d2 = MaxPool2D(padding='SAME')
        
        # Convolution 3
        self.conv3 = Conv2D(filters=256, kernel_size=3, activation=tf.nn.relu, padding='SAME')
        
        # Max pool 3
        self.maxpool2d3 = MaxPool2D(padding='SAME')
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dropout(0.2)
        self.d3 = Dense(10, activation='softmax')

    def call(self, x):
        # Convolution 1
        x = self.conv1(x)
        
        # Max pool 1
        x = self.maxpool2d1(x)
        
        # Convolution 2 
        x = self.conv2(x)
        
        # Max pool 2 
        x = self.maxpool2d2(x)
        
        # Convolution 3
        x = self.conv3(x)
        # Max pool 3
        x = self.maxpool2d3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        out = self.d3(x)
        return out

model = CNN_Model()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])



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


