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

# Mini-Imagenet dataset download from Auther's Github repository
import gdown

google_path = 'https://drive.google.com/uc?id='
file_id = '1TxNMS2lYPfY_uCZdLd_9jnFKT5l7WdwC'
output_name = 'imagenet_part_1.zip'
gdown.download(google_path+file_id,output_name,quiet=False)
# https://drive.google.com/file/d/1TxNMS2lYPfY_uCZdLd_9jnFKT5l7WdwC/view?usp=sharing

% rm -rf sample_data
!unzip /content/imagenet_part_1.zip -d /content/data
clear_output()
! rm /content/imagenet_part_1.zip

google_path = 'https://drive.google.com/uc?id='
file_id = '131M1Zr1ASj7V0Zty4yGMRrwHLYTUCqJO'
output_name = 'imagenet_part_2.zip'
gdown.download(google_path+file_id,output_name,quiet=False)
# https://drive.google.com/file/d/131M1Zr1ASj7V0Zty4yGMRrwHLYTUCqJO/view?usp=sharing

!unzip /content/imagenet_part_2.zip -d /content/data
clear_output()
! rm /content/imagenet_part_2.zip


# if you want to 200 classes, Activate below code

"""
google_path = 'https://drive.google.com/uc?id='
file_id = '1lxj8yuncbehS6V9IdGYkGJueeNZi800T'
output_name = 'imagenet_part_3.zip'
gdown.download(google_path+file_id,output_name,quiet=False)
# https://drive.google.com/file/d/1lxj8yuncbehS6V9IdGYkGJueeNZi800T/view?usp=sharing

!unzip /content/imagenet_part_3.zip -d /content/data
clear_output()
! rm /content/imagenet_part_3.zip

google_path = 'https://drive.google.com/uc?id='
file_id = '1YuF7pkuHxGLaEcYuCgvLEkcb2127NUGO'
output_name = 'imagenet_part_4.zip'
gdown.download(google_path+file_id,output_name,quiet=False)
# https://drive.google.com/file/d/1YuF7pkuHxGLaEcYuCgvLEkcb2127NUGO/view?usp=sharing

!unzip /content/imagenet_part_4.zip -d /content/data
clear_output()
! rm /content/imagenet_part_4.zip
"""



from glob import glob
import random
import os
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model, Sequential
import numpy as np

def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image

IMG_SIZE = 64

X_train, Y_train = [],[]
path = "/content/data/train"
classes = os.listdir(path)
filenames = glob(path + '/*/*')
random.shuffle(filenames)
X_train = [parse_image(name) for name in filenames]
Y_train = [classes.index(name.split('/')[-2]) for name in filenames]

X_train = np.array(X_train) 
Y_train = np.array(Y_train) 

X_test, Y_test = [],[]
path = "/content/data/test"
classes = os.listdir(path)
filenames = glob(path + '/*/*')
random.shuffle(filenames)
X_test = [parse_image(name) for name in filenames]
Y_test = [classes.index(name.split('/')[-2]) for name in filenames]

X_test = np.array(X_test) 
Y_test = np.array(Y_test) 

X_train, X_test = X_train / 255.0, X_test / 255.0

train_size = 50
test_size  = 100
STEPS = int(len(X_train)/train_size)
VAL_STEPS = int(len(X_test)/test_size)

# returns batch_size random samples from either training set or validation set
# resizes each image to (224, 244, 3), the native input size for VGG19
# Define network
IMG_SIZE = 160                      # Default is 224, use 160 due to Colab GPU memory
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = 100                    # mini-imagenet

import sys
from tensorflow.keras import layers

class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BuildResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes):
        super(BuildResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)

def ResNet(model_type, num_classes):
    if model_type == 'resnet18':
        return BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif model_type == 'resnet34':
        return BuildResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif model_type == 'resnet50':
        return BuildResNet(BottleNeck, [3, 4, 6, 3], num_classes)
    elif model_type == 'resnet101':
        return BuildResNet(BottleNeck, [3, 4, 23, 3], num_classes)
    elif model_type == 'resnet152':
        return BuildResNet(BottleNeck, [3, 8, 36, 3], num_classes)
    else:
        sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))

model = ResNet('resnet18', num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_name = 'cifar10_resnet18'

import cv2
def getBatch(batch_size, train_or_val='train'):
    x_batch = []
    y_batch = []
    if train_or_val == 'train':
        idx = np.random.randint(0, len(X_train), (batch_size))

        for i in idx:
            img = cv2.resize(X_train[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            x_batch.append(img)
            # x_batch.append(X_train[i])
            y_batch.append(Y_train[i])
    elif train_or_val == 'val':
        idx = np.random.randint(0, len(X_test), (batch_size))

        for i in idx:
            img = cv2.resize(X_test[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            x_batch.append(img)
            # x_batch.append(X_test[i])
            y_batch.append(Y_test[i]) 
    else:
        print("error, please specify train or val")

    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch

from tqdm import tqdm, tqdm_notebook, trange
EPOCHS = 3
for epoch in range(EPOCHS):

    with tqdm_notebook(total=STEPS, desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        for s in range(STEPS):
            x_batch, y_batch = getBatch(train_size, "train")
            out= model.train_on_batch(x_batch, y_batch)
            loss_val = out[0]
            acc      = out[1]*100

            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")
            
    with tqdm_notebook(total=VAL_STEPS, desc=f"Test_ Epoch {epoch+1}") as pbar:    
        test_losses = []
        test_accuracies = []
        for s in range(VAL_STEPS):
            x_batch_val, y_batch_val = getBatch(test_size, "val")
            evaluation = model.evaluate(x_batch_val, y_batch_val)
            
            loss_val= evaluation[0]
            acc     = evaluation[1]*100
            
            test_losses.append(loss_val)
            test_accuracies.append(acc)
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")

 