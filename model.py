import csv
import cv2
import numpy as np

lines = []
print('opening data/driving_logs.csv')
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("# lines: ", len(lines))

images = []
measurements = []
for line in lines[1:]:
    for i in range(3):
        if (i == 1): # left
            correction = 0.2
        elif (i == 2): # right
            correction = -0.2
        else: # center
            correction = 0.2
        source_path = line[i]
        filename = source_path.split('/')[-1]
        image = cv2.imread('data/IMG/{}'.format(filename))
        images.append(image)
        measurements.append(float(line[3]) + correction)

# augment images by flipping
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print (np.amax(X_train), y_train.shape)

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D

# %%

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# ignore top/bottom part of image
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
                           batch_size=128, nb_epoch=5)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

