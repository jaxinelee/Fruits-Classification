import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import seaborn as sb
import matplotlib.pyplot as plt
import cv2
import os
from keras.preprocessing.image import load_img,img_to_array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import time

train_images = []       
train_labels = []
shape = (64,64)  
train_path = '../input/trainingg'
for filename in os.listdir('../input/trainingg'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(train_path,filename))
        train_labels.append(filename.split('_')[0])   
        img = cv2.resize(img,shape)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_images.append(rgb_img)
train_labels = pd.get_dummies(train_labels).values #one hot encoding
train_images = np.array(train_images) #transform into array

test_images = []       
test_labels = []
shape = (64,64)  
test_path = '../input/testing'
for filename in os.listdir('../input/testing'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(test_path,filename))
        test_labels.append(filename.split('_')[0])
        img = cv2.resize(img,shape)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_images.append(rgb_img)
test_labels = pd.get_dummies(test_labels).values
test_images = np.array(test_images)

x_train = x_train / 255 
x_test = x_test / 255 #Dividing the data with 255 will normalize the pixel intensity values


model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",input_shape=(64,64,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(4, activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state = 1)
start = time.time()
history = model.fit(x_train, y_train, batch_size=64, epochs=16, verbose=1, validation_data = (X_valid,Y_valid), shuffle=True)
end = time.time()

score = model.evaluate(x_train, y_train, verbose=0)
print(f'Train loss: {score[0]} / Train accuracy: {score[1]}')

score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

print("--- %s seconds ---" % (end-start))

#summarize history for accuracy
plt.plot(history.history["accuracy"],color="black",label="Train Accuracy")
plt.plot(history.history["val_accuracy"],color="purple",label="Validation Accuracy")
plt.title("Accuracy Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy Values")
plt.show()

#summarize history for loss
plt.plot(history.history["loss"],color="blue",label="Train Loss")
plt.plot(history.history["val_loss"],color="red",label="Validation Loss")
plt.title("Loss Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss Values")
plt.show()
