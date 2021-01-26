# example of loading the fashion mnist dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Choose the dataset name

# Datasets = "cifar_10_32_pixels"
# Datasets = "cifar_100_32_pixels"
Datasets = "cifar_10_224_pixels"
# Datasets = "cifar_100_224_pixels"

if Datasets == "cifar_10_32_pixels":
    n_classes = 10
    img_size = 32

    dst_dir_train = './01_CIFAR10_32pixels/train/'
    dst_dir_test = './01_CIFAR10_32pixels/test/'
    
    # Load the CIFAR-10 dataset
    cifar10 = tf.keras.datasets.cifar10
    # Get test and training data where x are the images and y are the labels
    # load dataset
    (trainX, trainy), (testX, testy) = cifar10.load_data()

    CLASSES = ['airplane', 'automobile', 'bird', 'cat','deer',
               'dog','frog','horse','ship','truck']

elif Datasets == "cifar_100_32_pixels":
    n_classes = 100
    img_size = 32
    
    dst_dir_train = './02_CIFAR100_32pixels/train/'
    dst_dir_test = './02_CIFAR100_32pixels/test/'
    
    # Load the CIFAR-100 dataset
    cifar100 = tf.keras.datasets.cifar100
    # Get test and training data where x are the images and y are the labels
    # load dataset
    (trainX, trainy), (testX, testy) = cifar100.load_data()

    CLASSES = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm' ]
    
elif Datasets == "cifar_10_224_pixels":
    n_classes = 10
    img_size = 224

    dst_dir_train = './03_CIFAR10_224pixels/train/'
    dst_dir_test = './03_CIFAR10_224pixels/test/'
    
    # Load the CIFAR-10 dataset
    cifar10 = tf.keras.datasets.cifar10
    # Get test and training data where x are the images and y are the labels
    # load dataset
    (trainX, trainy), (testX, testy) = cifar10.load_data()

    CLASSES = ['airplane', 'automobile', 'bird', 'cat','deer',
               'dog','frog','horse','ship','truck']

elif Datasets == "cifar_100_224_pixels":
    n_classes = 100
    img_size = 224
    
    dst_dir_train = './04_CIFAR100_224pixels/train/'
    dst_dir_test = './04_CIFAR100_224pixels/test/'
    
    # Load the CIFAR-100 dataset
    cifar100 = tf.keras.datasets.cifar100
    # Get test and training data where x are the images and y are the labels
    # load dataset
    (trainX, trainy), (testX, testy) = cifar100.load_data()

    CLASSES = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm' ]
    
# os.mkdir(dst_dir_train)
# os.mkdir(dst_dir_test)
os.makedirs(dst_dir_train, exist_ok=True)
os.makedirs(dst_dir_test, exist_ok=True)

for folder in CLASSES:
    os.mkdir(os.path.join(dst_dir_train,folder))
    os.mkdir(os.path.join(dst_dir_test,folder))

#---------------------------------------------------------------------------

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
print(trainX.shape[0], testX.shape[0])

# import sys
# sys.exit()

# for idx in range (2000):
for idx in range (trainX.shape[0]):
    your_image = trainX[idx]
    your_image = cv2.resize(trainX[idx], (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2,2)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(your_image, aspect='auto')
    fig.savefig(dst_dir_train+str(CLASSES[int(trainy[idx])])+'/'+str(idx)+'.jpg',dpi=img_size/2)

    plt.close('all')
    if (idx+1) % 100 ==0:
        print(idx+1,"train images were converted and saved!")

# for idx in range (400):
for idx in range (testX.shape[0]):
    your_image = testX[idx]
    your_image = cv2.resize(testX[idx], (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2,2)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(your_image, aspect='auto')
    fig.savefig(dst_dir_test+str(CLASSES[int(testy[idx])])+'/'+str(idx)+'.jpg',dpi=img_size/2)

    plt.close('all')
    if (idx+1) % 100 ==0:
        print(idx+1,"test images were converted and saved!")


#---------------------------------------------------------------------------

# Function to rename multiple files 
def main():
    for idx in range(n_classes):
        for count, filename in enumerate(os.listdir(dst_dir_train+str(CLASSES[idx])+"/")): 
            dst = str(CLASSES[idx])+"_"+str(count).zfill(5) + ".jpg"
            src =dst_dir_train+str(CLASSES[idx])+"/"+ filename 
            dst =dst_dir_train+str(CLASSES[idx])+"/"+ dst 

            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 


    for idx in range(n_classes):
        for count, filename in enumerate(os.listdir(dst_dir_test+str(CLASSES[idx])+"/")): 
            dst = str(CLASSES[idx])+"_"+str(count).zfill(5) + ".jpg"
            src =dst_dir_test+str(CLASSES[idx])+"/"+ filename 
            dst =dst_dir_test+str(CLASSES[idx])+"/"+ dst 

            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 

 # Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 