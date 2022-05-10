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
from tensorflow.keras import Model, Sequential, layers

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
# Define network
IMG_SIZE = 224                      # VGG19
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = 10                    # cifar10

import math
import sys

class BottleNeck(tf.keras.Model):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(4*growth_rate, kernel_size=1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(growth_rate, kernel_size=3, padding='same', use_bias=False)
            
    def call(self, x):
        out = self.conv1(tf.keras.activations.relu(self.bn1(x)))
        out = self.conv2(tf.keras.activations.relu(self.bn2(out)))
        out = layers.concatenate([out, x])
        return out

class Transition(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = layers.BatchNormalization()
        self.conv = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=2)
        
    def call(self, x):
        out = self.conv(tf.keras.activations.relu(self.bn(x)))
        out = self.avg_pool2d(out)
        return out

class BuildDenseNet(tf.keras.Model):
    def __init__(self, block, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(BuildDenseNet, self).__init__()
        self.growth_rate = growth_rate
        
        num_channels = 2*growth_rate
        self.conv1 = layers.Conv2D(num_channels, kernel_size=3, padding='same', use_bias=False)
        
        self.dense1 = self._make_layer(block, num_channels, num_blocks[0])
        num_channels += num_blocks[0] * growth_rate
        out_channels = int(math.floor(num_channels*reduction))
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense2 = self._make_layer(block, num_channels, num_blocks[1])
        num_channels += num_blocks[1] * growth_rate
        out_channels = int(math.floor(num_channels*reduction))
        self.trans2 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense3 = self._make_layer(block, num_channels, num_blocks[2])
        num_channels += num_blocks[2] * growth_rate
        out_channels = int(math.floor(num_channels*reduction))
        self.trans3 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense4 = self._make_layer(block, num_channels, num_blocks[3])
        num_channels += num_blocks[3] * growth_rate
        
        self.bn = layers.BatchNormalization()
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = tf.keras.activations.relu(self.bn(self.dense4(out)))
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, in_channels, num_block):
        layer = []
        for i in range(num_block):
            layer += [block(in_channels, self.growth_rate)]
            in_channels += self.growth_rate
        return tf.keras.Sequential(layer)

def DenseNet(model_type, num_classes):
    if model_type == 'densenet121':
        return BuildDenseNet(BottleNeck, [6, 12, 24, 16], growth_rate=32, num_classes=num_classes)
    elif model_type == 'densenet161':
        return BuildDenseNet(BottleNeck, [6, 12, 36, 24], growth_rate=48, num_classes=num_classes)
    elif model_type == 'densenet169':
        return BuildDenseNet(BottleNeck, [6, 12, 32, 32], growth_rate=32, num_classes=num_classes)
    elif model_type == 'densenet201':
        return BuildDenseNet(BottleNeck, [6, 12, 48, 32], growth_rate=32, num_classes=num_classes)
    else:
        sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))


model = DenseNet('densenet121', num_classes)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_name = 'cifar10_DenseNet'


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


