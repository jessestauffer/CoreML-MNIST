from keras import models
from keras import layers
from keras.utils import np_utils
from keras.datasets import mnist

BATCH_SIZE = 256
NUM_EPOCHS = 10
IMG_WIDTH = IMG_HEIGHT = 28
NUM_CHANNELS = 1
NUM_DIGITS = 10

# load pre-shuffled MNIST data
(X_train, y_train), (X_val, y_val) = mnist.load_data()

# reshape data
X_train = X_train.reshape(X_train.shape[0], IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)
X_val = X_val.reshape(X_val.shape[0], IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

# normalize data
X_train /= 255
X_val /= 255

# convert labels to categorical
y_train = np_utils.to_categorical(y_train, NUM_DIGITS)
y_val = np_utils.to_categorical(y_val, NUM_DIGITS)

# build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_DIGITS, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_val, y_val), verbose=1)

# save the weights
model.save_weights("mnist-weights.h5")
