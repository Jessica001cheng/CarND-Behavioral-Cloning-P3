import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, MaxPooling2D, Cropping2D,Dropout
#from keras.callbacks import ModelCheckpoint, Callback

def preprocess_image(img):
    '''
        Method for preprocessing images: this method is the same used in drive.py, except this version uses
        BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
        received in RGB)
        '''
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return new_img

def generator_train_data(images_path, measurements, batch_size=128):
    '''
    generator model train data to read, preprocess, augment data. then yield to the model.
    '''
    X,y=([],[])
    images_path,measurements=shuffle(images_path,measurements)
    while True:
        for i in range(len(images_path)):
            measurement = measurements[i]
            image = cv2.imread(images_path[i])
            if image is None:
                #print("file:{} can not load".format(filename))
                continue
            image=preprocess_image(image)
            X.append(image)
            y.append(measurements[i])
            if len(X) == batch_size:
                print("yield size: ",  len(X))
                yield (np.array(X), np.array(y))
                X,y=([],[])
                images_path,measurements=shuffle(images_path,measurements)
            ### append augment image also
            X.append(cv2.flip(image,1))
            y.append(measurement* -1.0)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X,y=([],[])
                images_path,measurements=shuffle(images_path,measurements)


'''
Main function
'''

lines=[]
csv_path_prepend = ["./data/", "./data-2ndtrack/"]
images_path=[]
images=[]
measurements=[]

for j in range(2):
    csv_path = csv_path_prepend[j] + "driving_log.csv"
    print("csv path: ", csv_path)
    with open(csv_path) as csvfile:
        reader=csv.reader(csvfile, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE)
        for line in reader:
            # skip it if ~0 speed - not representative of driving behavior
            if float(line[6]) < 0.1 :
                continue
            source_path=line[0]
            filename=source_path.split('/')[-1]
            curr_path=csv_path_prepend[j] +"IMG/" + filename
            images_path.append(curr_path)
            measurement = float(line[3])
            measurements.append(measurement)

## display the augment image
index = 0
image = cv2.imread(images_path[index])
print("file:{} will be augmented".format(images_path[index]))
augment_img = cv2.flip(image,1)
cv2.imwrite("./examples/Augment_image.jpg", augment_img)


## Model start here
input_shape = (160,320,3)
model = Sequential()
##cropping image
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=input_shape))
## normalized image
model.add(Lambda(lambda x: x / 255.0 - 0.5))
##add conv2D 5*5, 24 channel
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu'))
##add conv2D 5*5, 36 channel
model.add(Conv2D(36, kernel_size=(5,5),  strides=(2,2), activation='relu'))
## add conv2D 5*5, 48 channel
model.add(Conv2D(48, kernel_size=(5,5),  strides=(2,2), activation='relu'))
## add conv2d without pooling.
model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
## add conv2d without pooling.
model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="Adam", loss="mse")

batch_size = 128
train_gen = generator_train_data(images_path, measurements, batch_size)
val_gen = generator_train_data(images_path, measurements, batch_size)
#model.fit(X_train, y_train, validation_split=0.2, epochs=3,shuffle=True)
## ??? How to set the validatoin_steps and step_per_epoch
## as when reading image, there are some image discard. How can i know the total number of data in each epoch
real_image_size = 24036
train_size = int(real_image_size * 0.8)
val_size = int(real_image_size * 0.2)
print("train_size", train_size)
print("val_size", val_size)

## Use model generator to reduce memory usage
history = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=val_size//batch_size, steps_per_epoch=train_size//batch_size,epochs=3, verbose=1)

print(model.summary())

model.save("model.h5")
