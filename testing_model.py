# -*- coding: utf8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import numpy as np
import sys


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', input_shape=(32, 32, 3))) # 30x30
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid')) # 28x28
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 14x14
model.add(Dropout(0.25))


model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', input_shape=(32, 32, 3))) # 12x12
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid')) # 10x10
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 5x5
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(2, 2), padding='valid')) # 4x4
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Conv2D(128, kernel_size=(2, 2), padding='valid')) # 3x3
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 2x2
model.add(Dropout(0.25))  

model.add(Flatten())

model.add(Dense(128, activation='selu', kernel_initializer='lecun_uniform'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='selu', kernel_initializer='lecun_uniform'))
#model.add(Dropout(0.5))
#model.add(Dense(128, activation='selu', kernel_initializer='lecun_uniform'))

model.add(Dense(10, kernel_initializer='lecun_uniform'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

try:
	model.load_weights(sys.argv[1])
	opt = Adam(lr=0.005, decay=1e-4)
	model.compile(loss='categorical_crossentropy',
								optimizer=opt,
								metrics=['accuracy'])

except IOError:
	print "IOError: não é possível abrir o arquivo", sys.argv[1]
	exit(1)

except:
	print "Unknown Error: algo deu errado"
	exit(1)

# Load the dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Evaluate the model
#scores = model.evaluate(X_test, to_categorical(Y_test))
#print('Accuracy: %.3f' % scores[1])

Y_pred = np.argmax(model.predict(X_test), axis=1)
print Y_pred.shape, Y_test.shape
print classification_report(Y_test, Y_pred)