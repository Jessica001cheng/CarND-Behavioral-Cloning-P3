import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

lines=[]
with open("./data/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images=[]
measurements=[]
for line in lines:
    source_path=line[0]
    filename=source_path.split('/')[-1]
    curr_path="./data/IMG/" + filename
    #print("curr_path: ", curr_path)
    image = cv2.imread(curr_path)

    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train=np.array(images)
y_train=np.array(measurements)
print("X train shape", X_train.shape)
print("y train shape", y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, MaxPooling2D

input_shape = (160,320,3)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
##add conv2D 5*5, 6 channel
model.add(Conv2D(6, kernel_size=(5,5), padding='valid', activation='relu'))
## add MaxPooling2D layer, 2*2s
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
##add conv2D 5*5, 6 channel
model.add(Conv2D(6, kernel_size=(5,5), padding='valid', activation='relu'))
## add MaxPooling2D layer, 2*2s
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer="Adam", loss="mse")
model.fit(X_train, y_train, validation_split=0.2, epochs=2,shuffle=True)

model.save("model.h5")