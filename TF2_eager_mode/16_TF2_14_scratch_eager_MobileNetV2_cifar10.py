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
# Define network
IMG_SIZE = 224                      # VGG19
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = 10                    # cifar10

class Block(tf.keras.Model):
    '''Expand + depthwise & pointwise convolution'''
    def __init__(self, in_channels, out_channels, expansion, strides):
        super(Block, self).__init__()
        self.strides = strides
        channels = expansion * in_channels
        
        self.conv1 = layers.Conv2D(channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(channels, kernel_size=3, strides=strides, padding='same',
                                   groups=channels, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides == 1 and in_channels != out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=1, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
        
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out]) if self.strides==1 else out
        return out

class MobileNetV2(tf.keras.Model):
    # (expansion, out_channels, num_blocks, strides)
    config = [(1, 16, 1, 1),
              (6, 24, 2, 1),  # NOTE: change strides 2 -> 1 for CIFAR10
              (6, 32, 3, 2),
              (6, 64, 4, 2),
              (6, 96, 3, 1),
              (6, 160, 3, 2),
              (6, 320, 1, 1)]
    
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 strides 2 -> 1 for CIFAR10
        self.conv1 = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer = self._make_layers(in_channels=32)
        self.conv2 = layers.Conv2D(1280, kernel_size=1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer(out)
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
        
    def _make_layers(self, in_channels):
        layer = []
        for expansion, out_channels, num_blocks, strides in self.config:
            stride = [strides] + [1]*(num_blocks-1)
            for s in stride:
                layer += [Block(in_channels, out_channels, expansion, s)]
                in_channels = out_channels
        return tf.keras.Sequential(layer)

model = MobileNetV2(num_classes)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_name = 'cifar10_MobileNetV2'


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


