import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Lambda, Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

# import simulator data
lines, angles = [], []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if float(line[6]) < 0.1:  # omit angles while stopped 
            continue
        lines.append(line)
        angles.append(float(line[3]))

# initial histogram of steering angles
nbins = 15
hrange = (-0.5,0.5)
hist, bins = np.histogram(angles, nbins, range=hrange)
(a_min, a_max) = bins[int(nbins/2)], bins[int(nbins/2)+1]
samples = list(zip(lines, angles))
np.random.shuffle(samples)

# center bin at 0.0 is overrepresented, it will be reduced
filtered_lines, filtered_angles = [], []
nlimit = int(hist[int(nbins / 2)-1] * 1.1)
ncount = 0
for line,angle in samples:
    if angle > a_min and angle < a_max:
        ncount += 1
        if ncount > nlimit:
            continue
    filtered_lines.append(line)
    filtered_angles.append(angle)

# left, center and right camera image filenames are extracted
# left and right angles are compensated by 0.25 
augmented_lines, augmented_angles = [], []
for line in filtered_lines:
    for source_path, correction in zip([line[0],line[1],line[2]], [0,0.25,-0.25]):
        filename = 'data/IMG/' + source_path.split('/')[-1]
        angle = float(line[3]) + correction
        augmented_lines.append(filename)
        augmented_angles.append(angle)

# samples are filename/angle pairs, split to training and validation sets
samples = list(zip(augmented_lines, augmented_angles))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/' + batch_sample[0].split('/')[-1]
                # basic preprocessing is performed
                image = cv2.imread(name)
                # top is cropped just below horizon, bottom car hood is also unnecessary
                image = image[70:140,:,:]
                # resized image still contains enough information
                image = cv2.resize(image,(80, 35), interpolation = cv2.INTER_AREA)
                # YUV color space is advised to be used in NVidia paper
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

batch_size = 64
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# CNN model similar, but simplified version of the one used in Nvidia paper
model = Sequential()
# image normalization as usual
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (35, 80, 3)))
model.add(Convolution2D(24, (5, 5), strides=(2,2), activation="elu"))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation="elu"))
model.add(Convolution2D(48, (5, 5), activation="elu"))
model.add(Flatten())
# aggressive dropout
model.add(Dropout(0.2))
model.add(Dense(100, activation="elu"))
model.add(Dropout(0.6))
model.add(Dense(50, activation="elu"))
model.add(Dropout(0.8))
model.add(Dense(10, activation="elu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=0.0002))
model.fit_generator(train_generator, steps_per_epoch=int(len(train_samples)/batch_size), validation_data=validation_generator,
            validation_steps=int(len(validation_samples)/batch_size), epochs=5, verbose=1)
model.save('model.h5')
print(model.summary())

model.save_weights('model_weights.h5')
json_string = model.to_json()
with open('model.json', 'w') as f:
    f.write(json_string)
