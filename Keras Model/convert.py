import coremltools
from keras import models
from keras import layers

IMG_WIDTH = IMG_HEIGHT = 28
NUM_CHANNELS = 1
NUM_DIGITS = 10

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

# load the saved weights
model.load_weights("mnist-weights.h5")

output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# convert the model to a CoreML model
coreml_model = coremltools.converters.keras.convert(
	model,
	input_names=['image'],
	image_input_names=['image'],
	output_names='output',
	class_labels=output_labels,
	image_scale=1/255.0
)

# save the CoreML model
coreml_model.save('MNISTModel.mlmodel')
