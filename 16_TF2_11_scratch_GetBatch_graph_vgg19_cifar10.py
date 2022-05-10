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

IMG_SIZE = 128

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

train_size = 100
test_size  = 200
STEPS = int(len(X_train)/train_size)
VAL_STEPS = int(len(X_test)/test_size)


from tensorflow.keras import layers
# returns batch_size random samples from either training set or validation set
# resizes each image to (224, 244, 3), the native input size for VGG19
# Define network
IMG_SIZE = 224                      # VGG19
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = 10                    # cifar10

config = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(tf.keras.Model):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.conv = self._make_layers(config[vgg_name])
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layers(self, config):
        layer = []
        for l in config:
            if l == 'M':
                layer += [layers.MaxPool2D(pool_size=2, strides=2)]
            else:
                layer += [layers.Conv2D(l, kernel_size=3, padding='same'),
                          layers.BatchNormalization(),
                          layers.ReLU()]
        layer += [layers.AveragePooling2D(pool_size=1, strides=1)]
        return tf.keras.Sequential(layer)

model = VGG("vgg11", num_classes)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_name = 'cifar10_VGG19'

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

 