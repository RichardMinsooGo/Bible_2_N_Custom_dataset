import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model, Sequential

batch_size = 128
n_epoch = 5

# path joining version for other paths

Datasets = "cifar_10_32_pixels"
# Datasets = "cifar_100_32_pixels"
# Datasets = "cifar_10_224_pixels"
# Datasets = "cifar_100_224_pixels"
# Datasets = "mini_imagenet"

if Datasets == "cifar_10_32_pixels":
    n_classes = 10
    img_size = 32

    dst_dir_train = './01_CIFAR10_32pixels/train/'
    dst_dir_test = './01_CIFAR10_32pixels/test/'
    
elif Datasets == "cifar_100_32_pixels":
    n_classes = 100
    img_size = 32
    
    dst_dir_train = './02_CIFAR100_32pixels/train/'
    dst_dir_test = './02_CIFAR100_32pixels/test/'
    
elif Datasets == "cifar_10_224_pixels":
    n_classes = 10
    img_size = 224

    dst_dir_train = './03_CIFAR10_224pixels/train/'
    dst_dir_test = './03_CIFAR10_224pixels/test/'
    
elif Datasets == "cifar_100_224_pixels":
    n_classes = 100
    img_size = 224
    
    dst_dir_train = './04_CIFAR100_224pixels/train/'
    dst_dir_test = './04_CIFAR100_224pixels/test/'
    
elif Datasets == "mini_imagenet":
    n_classes = 200
    img_size = 160
    
    dst_dir_train = './07_mini_imagenet/train/'
    dst_dir_test = './07_mini_imagenet/test/'
    
    
# DIR = '/tmp'
num_train = sum([len(files) for r, d, files in os.walk(dst_dir_train)])
num_test  = sum([len(files) for r, d, files in os.walk(dst_dir_test)])

steps_per_epoch = int(num_train/batch_size)
validation_steps = int(num_test/batch_size)

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape = (img_size, img_size, 3), activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPool2D(pool_size = (2, 2)))

# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = n_classes, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(dst_dir_train,
                                                 target_size = (img_size, img_size),
                                                 batch_size = batch_size,
                                                 # class_mode = 'sparse')
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(dst_dir_test,
                                            target_size = (img_size, img_size),
                                            batch_size = batch_size,
                                            # class_mode = 'sparse')
                                            class_mode = 'categorical')

model.fit_generator(training_set,steps_per_epoch = steps_per_epoch, epochs = n_epoch, 
                    validation_data = test_set, validation_steps = validation_steps)



