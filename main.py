import numpy as np
import pandas as pd
import tensorflow
import os
import random
import shutil

for dirname, _, filenames in os.walk('/home/surya/Documents/Internship/archive'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

dataset_path = "/home/surya/Documents/Internship/archive"

labels = os.listdir(dataset_path)
print(labels)

for label in labels:
    print(os.listdir(os.path.join(dataset_path, label)))


output_path = '/home/surya/Documents/Internship/water_bottle_dataset_transformed'

if not os.path.exists(output_path):
    os.mkdir(os.path.join(output_path))

for label in labels:
    sub_label = os.path.join(dataset_path,label)
    sub_labels_path = os.path.join(sub_label, label)
    label_path = os.listdir(sub_labels_path)
    ims = [i for i in os.listdir(sub_labels_path) if i.endswith(".jpeg")]
    random.shuffle(ims)
    split_size = 0.8
    train_len = int(len(ims) * split_size)
    train_ims = ims[:train_len]
    val_ims = ims[train_len:]
    
    # create train and val dirs
    train_path = os.path.join(output_path, "train")
    label_train_path = os.path.join(train_path, label)
    

    val_path = os.path.join(output_path, "val")
    label_val_path = os.path.join(val_path, label)
    
    if not os.path.exists(train_path):
        os.mkdir(train_path)
        
    if not os.path.exists(label_train_path):
        os.mkdir(label_train_path)
        
    if not os.path.exists(val_path):
        os.mkdir(val_path)  
    
    if not os.path.exists(label_val_path):
        os.mkdir(label_val_path)
        
    for im in train_ims:
        shutil.copy(os.path.join(sub_labels_path, im), label_train_path)
    
    for im in val_ims:
        shutil.copy(os.path.join(sub_labels_path, im), label_val_path)

# path = '/home/surya/Documents/Internship/water_bottle_dataset_transformed'
# shutil.rmtree(path)

def TotalImages():
    src = os.listdir(output_path)
    
    for lb  in src:
        print("\nDataset Split Type: ", lb)
        x = os.path.join(output_path, lb)
        y = os.listdir(x)

        for sb_labels in y:
            z = os.path.join(x, sb_labels)
            print("Total images in ", sb_labels, " is:\t", len(os.listdir(z)))

TotalImages()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import io
import numpy as np
import os
from PIL import Image

datagen = ImageDataGenerator(        
            width_shift_range=0.1,  
            height_shift_range=0.1,    
            brightness_range = (0.3, 0.9),
            zoom_range=0.2)


for label in labels:
    image_directory = train_path + '/' + label + '/'
    SIZE = 150
    dataset = []

    print(image_directory)
    my_images = os.listdir(image_directory)
    for i, image_name in enumerate(my_images):    
        if ((image_name.split('.')[1] == 'jpeg')):
            image = load_img(image_directory + image_name, target_size = (150,150))
            image = img_to_array(image)
            dataset.append(image)

    x = np.array(dataset)
    i = 0
    for batch in datagen.flow(x, batch_size=16,
                            save_to_dir= train_path + '/' + label + '/',
                            save_prefix='aug',
                            save_format='jpeg'):
        i += 1    
        if i > 50:        
            break

for label in labels:
    image_directory = val_path + '/' + label + '/'
    SIZE = 150
    dataset = []

    print(image_directory)
    my_images = os.listdir(image_directory)
    for i, image_name in enumerate(my_images):    
        if ((image_name.split('.')[1] == 'jpeg')):
            image = load_img(image_directory + image_name, target_size = (150,150))
            image = img_to_array(image)
            dataset.append(image)

    x = np.array(dataset)
    i = 0
    for batch in datagen.flow(x, batch_size=16,
                            save_to_dir= val_path + '/' + label + '/',
                            save_prefix='aug',
                            save_format='jpeg'):
        i += 1    
        if i > 20:        
            break

TotalImages()

import pathlib
import tensorflow as tf

train_dataset_url = train_path
train_data_dir = pathlib.Path(train_dataset_url)

validation_dataset_url = val_path
validation_data_dir = pathlib.Path(validation_dataset_url)

img_height,img_width= 150,150
batch_size=16
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  validation_data_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(6):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    layers.Conv2D(32, (5,5), activation='relu', padding= 'valid', input_shape=(150,150,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    BatchNormalization(),
    
    layers.Conv2D(32, (5,5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    BatchNormalization(),
    
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    BatchNormalization(),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    BatchNormalization(),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax'),
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=30)


fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train'])
plt.show()

