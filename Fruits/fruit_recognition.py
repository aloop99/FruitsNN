import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator

batch_size = 16
num_train_samples = 41322
num_validation_samples = 13877

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'fruits-360\Training',
        target_size=(100, 100),
        batch_size=batch_size,
        class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(
        'fruits-360\Test',
        target_size=(100, 100),
        batch_size=batch_size,
        class_mode='sparse')

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(81, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=num_validation_samples // batch_size)
model.save('my_model.h5')
