# Build the Image classification model by dividing the model into following 4 stages:
# a. Loading and preprocessing the image data
# b. Defining the model’s architecture
# c. Training the model
# d. Estimating the model’s performance

import zipfile

!wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

zip_ref=zipfile.ZipFile("pizza_steak.zip")
zip_ref.extractall()
zip_ref.close()

#inspect data 
!ls pizza_steak/train/

!ls pizza_steak/train/steak

import os

for dirpath,dirnames,filenames in os.walk("pizza_steak"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")

num_steak_images_train=len(os.listdir("pizza_steak/train/steak"))

num_steak_images_train

#get the classnames programmtically
import pathlib
import numpy as np
data_dir=pathlib.Path("pizza_steak/train")
class_names=np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)

#visualize image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir,target_class):
  target_folder=target_dir+target_class

  #get random image path
  random_image = random.sample(os.listdir(target_folder), 1)
  print(random_image)

  img=mpimg.imread(target_folder + "/" +random_image[0])
  plt.imshow(img)
  plt.title(target_class)

  plt.axis("off")
  print(img.shape)

  return img

img=view_random_image(target_dir="pizza_steak/train/",target_class="pizza")

img

import tensorflow as tf
tf.constant(img)

img.shape # returns width ,height, color channels

#normalize image  i.e. turn between 0 and 1from
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)

train_datagen=ImageDataGenerator(rescale=1./255)
valid_datagen=ImageDataGenerator(rescale=1./255)

train_dir="/content/pizza_steak/train"
test_dir="/content/pizza_steak/test"

train_data=train_datagen.flow_from_directory(directory=train_dir,
                                             batch_size=32,
                                             target_size=(224,224),
                                             class_mode="binary",
                                             seed=42)

valid_data=valid_datagen.flow_from_directory(directory=test_dir,
                                             batch_size=32,
                                             target_size=(224,224),
                                             class_mode="binary",
                                             seed=42)

model_1=tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=10,kernel_size=3,
                                                           activation="relu",
                                                           input_shape=(224,224,3)),
                                    
                                    tf.keras.layers.Conv2D(10,3,activation="relu"),
                                    tf.keras.layers.MaxPool2D(pool_size=2, padding="valid"),
                                    
                                    tf.keras.layers.Conv2D(10,3,activation="relu"),
                                    tf.keras.layers.Conv2D(10,3,activation="relu"),
                                    tf.keras.layers.MaxPool2D(2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1,activation="sigmoid")
                                    ])

# compile CNN

model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# #fit the model
history_1=model_1.fit(train_data,
                       epochs=5,
                       steps_per_epoch=len(train_data), 
                       validation_data=valid_data,
                       validation_steps=len(valid_data))
