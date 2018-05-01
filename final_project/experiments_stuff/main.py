from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras import regularizers, optimizers
import os

batch_size = 32
num_classes = 10
epochs = 50
model_path = os.path.join(os.getcwd(), 'advanced_cifar10_model_0.h5') 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train[:5000]
# y_train = y_train[:5000]
# x_test = x_test[:1000]
# y_test = y_test[:1000] 

 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def SimpleCNN():
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	return model

def AdvancedCNN(baseMapNum = 32, weight_decay = 1e-4):
	model = Sequential()

	model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.3))

	model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.4))

	model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax'))

	model.summary()
	return model

def evaluate(to_load_model):
	model = load_model(to_load_model)
	scores = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss: {}\tTest accuracy: {}'.format(scores[0], scores[1]))
	K.clear_session()

def train(to_load_model = None):
	if to_load_model:
		model = load_model(to_load_model)
	else:
		model = AdvancedCNN()
	opt = optimizers.rmsprop(lr=0.0003, decay=1e-6)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
	mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[earlyStopping, mcp_save], validation_data=(x_test, y_test), shuffle=True)
	# model.save(model_path)

if __name__ == "__main__":
	# model = getModel()
	# train(os.path.join(os.getcwd(), 'cifar10_model_2.h5'))
	# train(os.path.join(os.getcwd(), 'advanced_cifar10_model_0.h5') )
	evaluate(os.path.join(os.getcwd(), 'advanced_cifar10_model_0.h5'))