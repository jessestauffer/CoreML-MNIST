from keras import models
from keras import layers
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 1024
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

# use image augmentation
train_gen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.08,
	shear_range=0.3,
	height_shift_range=0.08,
	zoom_range=0.8
)

val_gen = ImageDataGenerator()

train_generator = train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_gen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(NUM_DIGITS, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=train_generator.n//BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=val_generator, validation_steps=val_generator.n//BATCH_SIZE, verbose=1)

# save the weights
model.save_weights("mnist-weights.h5")
