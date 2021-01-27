import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, BatchNormalization
from tensorflow.keras import Input, Model, Sequential

from tensorflow.keras.utils import to_categorical
import numpy as np

import os, random
import numpy as np
import cv2
# import PIL
from PIL import Image

import sys

IMG_SIZE = 32
train_size = 20856

X_train = []
Y_train_age = []
Y_train_gender = []
Y_train_race = []

count = 0
# for idx in range(train_size):
    # random_file = random.choice(os.listdir("./train")) #change dir name to whatever
for random_file in sorted(os.listdir("./train")):
    
    count += 1
    # print("%d} %s"%(idx+1,random_file))
    if count % 50 == 0:
        print("  %s"%(random_file))
    
    if (random_file.find('__') != -1):
        print("Filename has error!!")
        sys.exit()
    
    age, gender, race, _ = random_file.split('_')
    gender = int(gender)
    if gender > 1:
        print("Filename has error!!")
        sys.exit()
        
    image = cv2.imread("./train/"+random_file)
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    img = np.array(img)/255.
    age = float(age)/116.
    
    X_train.append(img)
    Y_train_age.append(age)
    Y_train_gender.append(gender)
    Y_train_race.append(race)
    
    
    # print(age, gender, race, np.max(img))
    # print(img.shape)
print(np.max(Y_train_gender))

X_train = np.array(X_train)
Y_train_age = np.array(Y_train_age)
Y_train_gender = np.array(Y_train_gender)
Y_train_gender = to_categorical(Y_train_gender, 2)    

Y_train_race = np.array(Y_train_race)
Y_train_race = to_categorical(Y_train_race, 5)    

print(X_train.shape)
print(Y_train_age.shape)
print(Y_train_gender.shape)
print(Y_train_race.shape)

X_test = []
Y_test_age = []
Y_test_gender = []
Y_test_race = []

count = 0
# for idx in range(test_size):
    # random_file = random.choice(os.listdir("./test")) #change dir name to whatever
for random_file in sorted(os.listdir("./test")):
    
    # print("%d} %s"%(idx+1,random_file))
    count += 1
    if count % 50 == 0:
        print("  %s"%(random_file))
    
    if (random_file.find('__') != -1):
        print("Filename has error!!")
        sys.exit()
    
    age, gender, race, _ = random_file.split('_')
    gender = int(gender)
    if gender > 1:
        print("Filename has error!!")
        sys.exit()
        
    image = cv2.imread("./test/"+random_file)
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    img = np.array(img)/255.
    age = float(age)/116.
    
    X_test.append(img)
    Y_test_age.append(age)
    Y_test_gender.append(gender)
    Y_test_race.append(race)
    
    
    # print(age, gender, race, np.max(img))
    # print(img.shape)
print(np.max(Y_test_gender))

X_test = np.array(X_test)
Y_test_age = np.array(Y_test_age)
Y_test_gender = np.array(Y_test_gender)
Y_test_gender = to_categorical(Y_test_gender, 2)    

Y_test_race = np.array(Y_test_race)
Y_test_race = to_categorical(Y_test_race, 5)    


# import sys
# sys.exit()
    
LABELS_RACE = [
    'white', 
    'black',
    'asian',
    'indian',
    'others'
]
    
LABELS_GENDER = [
    'male', 
    'female'
]
        
class UtkMultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains three branches, one for age, other for 
    sex and another for race. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """
    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:
        
        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPool2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        return x

    def build_race_branch(self, inputs, num_races):
        """
        Used to build the race branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_races)(x)
        x = Activation("softmax", name="race_output")(x)

        return x

    def build_gender_branch(self, inputs, num_genders=2):
        """
        Used to build the gender branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        # x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_genders)(x)
        x = Activation("softmax", name="gender_output")(x)

        return x

    def build_age_branch(self, inputs):   
        """
        Used to build the age branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.

        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("linear", name="age_output")(x)

        return x

    def assemble_full_model(self, width, height, num_races):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)

        inputs = Input(shape=input_shape)

        age_branch = self.build_age_branch(inputs)
        race_branch = self.build_race_branch(inputs, num_races)
        gender_branch = self.build_gender_branch(inputs)

        model = Model(inputs=inputs,
                     outputs = [age_branch, race_branch, gender_branch],
                     name="face_net")

        return model
    
model = UtkMultiOutputModel().assemble_full_model(IMG_SIZE, IMG_SIZE, num_races=5)    

model.summary()

# init_lr = 1e-4
epochs = 5

# opt = Adam(lr=init_lr, decay=init_lr / epochs)

model.compile(optimizer='adam', 
              loss={
                  'age_output': 'mse', 
                  'race_output': 'categorical_crossentropy', 
                  'gender_output': 'categorical_crossentropy'},
              loss_weights={
                  'age_output': 4., 
                  'race_output': 1.5, 
                  'gender_output': 0.1},
              metrics={
                  'age_output': 'mae', 
                  'race_output': 'accuracy',
                  'gender_output': 'accuracy'})

model.fit(X_train, [Y_train_age, Y_train_race, Y_train_gender], epochs=epochs, batch_size = 256)

model.evaluate(X_test,  [Y_test_age, Y_test_race, Y_test_gender], verbose=1)              


"""
callbacks=callbacks, validation_data=valid_gen, validation_steps=len(valid_idx)//valid_batch_size)
X_train = []
Y_train_age = []
Y_train_gender = []
Y_train_race = []
"""


