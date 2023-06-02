'''
Data Engineering
'''

'''
D0. Custom dataset download from Auther's Google Drive
'''

!pip install --upgrade --no-cache-dir gdown

import gdown
from IPython.display import clear_output 

'''
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo/5_TF2_UCF101_video_classification.git
! git pull origin master
# ! git pull origin main
'''

# Mini-Imagenet dataset download from Auther's Github repository
google_path = 'https://drive.google.com/uc?id='
file_id = '1TxNMS2lYPfY_uCZdLd_9jnFKT5l7WdwC'
output_name = 'imagenet_part_1.zip'
gdown.download(google_path+file_id,output_name,quiet=False)
# https://drive.google.com/file/d/1TxNMS2lYPfY_uCZdLd_9jnFKT5l7WdwC/view?usp=sharing

! rm -rf sample_data
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


'''
D1. Import Libraries for Data Engineering
'''

# clear_output()
from glob import glob
import random
import os
import tensorflow as tf
import numpy as np

'''
D2. Image Parsing
'''

# Define parsing image function
def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image

IMG_SIZE = 64

# Train data parsing
X_train, Y_train = [],[]
path = "/content/data/train"
classes = os.listdir(path)
filenames = glob(path + '/*/*')
random.shuffle(filenames)
X_train = [parse_image(name) for name in filenames]
Y_train = [classes.index(name.split('/')[-2]) for name in filenames]

# Convert to numpy array
X_train = np.array(X_train) 
Y_train = np.array(Y_train) 

# Test data parsing
X_test, Y_test = [],[]
path = "/content/data/test"
classes = os.listdir(path)
filenames = glob(path + '/*/*')
random.shuffle(filenames)
X_test = [parse_image(name) for name in filenames]
Y_test = [classes.index(name.split('/')[-2]) for name in filenames]

# Convert to numpy array
X_test = np.array(X_test) 
Y_test = np.array(Y_test) 


# Count output dimension
APP_FOLDER = '/content/data/train'
totalFiles = 0
output_dim = 0

for base, dirs, files in os.walk(APP_FOLDER):
    print('Searching in : ',base)
    for directories in dirs:
        output_dim += 1
    for Files in files:
        totalFiles += 1

print("output_dim : ", output_dim)

'''
D3. Data Preprocessing
'''
# Normalizing
X_train, X_test = X_train / 255.0, X_test / 255.0

print(Y_train[0:10])
print(X_train.shape)

# One-Hot Encoding
# from tensorflow.keras.utils import to_categorical

# Y_train = to_categorical(Y_train, output_dim)
# Y_test = to_categorical(Y_test, output_dim)

'''
D4. EDA(? / Exploratory data analysis)
'''
import matplotlib.pyplot as plt

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import Input
import numpy as np

'''
M2. Set Hyperparameters
'''

# returns batch_size random samples from either training set or validation set
# resizes each image to (224, 244, 3), the native input size for VGG19
IMG_SIZE = 224                      # VGG19
hidden_size = 256

EPOCHS = 5
learning_rate = 0.001

'''
M3. Build NN model
'''
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
        self.d3 = Dense(output_dim, activation='softmax')

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
temp_inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
model(temp_inputs)
model.summary()

'''
M4. Optimizer
'''
# Optimizer can be included at model.compile

'''
M5. Model Compilation - model.compile
'''

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_name = 'cifar10_CNN'

'''
M6. Load trained model
'''

import os.path
if os.path.isfile(model_name+'.h5'):
    model.load_weights(model_name+'.h5')

'''
M7. Define getBatch Function for "model.train_on_batch"
'''
train_size = 50
test_size  = 100
STEPS = int(len(X_train)/train_size)
VAL_STEPS = int(len(X_test)/test_size)

import cv2

def getBatch(batch_size, train_or_val='train'):
    x_batch = []
    y_batch = []
    if train_or_val == 'train':
        idx = np.random.randint(0, len(X_train), (batch_size))

        for i in idx:
            img = cv2.resize(X_train[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            x_batch.append(img)
            y_batch.append(Y_train[i])
    elif train_or_val == 'val':
        idx = np.random.randint(0, len(X_test), (batch_size))

        for i in idx:
            img = cv2.resize(X_test[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            x_batch.append(img)
            y_batch.append(Y_test[i]) 
    else:
        print("error, please specify train or val")

    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch

# checkpoint was not used in this implementation
checkpoint = tf.train.Checkpoint(cnn=model)

'''
M8. Define Episode / each step process
'''

from tqdm import tqdm, tqdm_notebook, trange

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
            
'''
M9. Model evaluation
'''
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
        
'''
M10. Save Model
'''
model.save_weights(model_name+'.h5', overwrite=True)
 
