# -*- coding: utf8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical


from sklearn.model_selection import StratifiedKFold as KFold

# Load the dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# For reproducibility
np.random.seed(1000)
kfold = KFold(n_splits=3)

k_models = []
fold = 1
for train_idx, val_idx in kfold.split(X_train, Y_train):
	# Create the model
	model = Sequential()


	model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', input_shape=(32, 32, 3))) # 30x30
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu')) # 28x28
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) # 14x14
	model.add(Dropout(0.25))


	model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', input_shape=(32, 32, 3))) # 12x12
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu')) # 10x10
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) # 5x5
	model.add(Dropout(0.25))

	model.add(Conv2D(512, kernel_size=(2, 2), padding='valid', activation='relu')) # 4x4
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) # 2x2
	model.add(Dropout(0.25))  

	model.add(Flatten())
	model.add(Dense(1024, activation='selu', kernel_initializer='lecun_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='selu', kernel_initializer='lecun_uniform'))

	model.add(Dense(10, activation='softmax', kernel_initializer='lecun_uniform'))

	#opt = RMSprop(lr=0.001, decay=1e-9)
	#opt = Adagrad(lr=0.001, decay=1e-6)
	#opt = Adadelta(lr=0.075, decay=1e-6)
	opt = Adam(lr=0.0001, decay=1e-6)
	# Compile the model
	model.compile(loss='categorical_crossentropy',
								optimizer=opt,
								metrics=['accuracy'])

	checkpoint = ModelCheckpoint('saved_models/logs.{epoch:002d}--{val_loss:.2f}.hdf5', save_best_only=True)
	# Train the model
	model.fit(X_train[train_idx], to_categorical(Y_train[train_idx]),
						batch_size=128,
						shuffle=True,
						epochs=250,
						validation_data=(X_train[val_idx], to_categorical(Y_train[val_idx])),
						callbacks=[EarlyStopping(min_delta=0.001, patience=5), CSVLogger('training_' + 'fold_' + str(fold) + '.log', separator=',', append=False), checkpoint])
	k_models.append(model)
	fold += 1
# Evaluate the model
#scores = model.evaluate(X_test, to_categorical(Y_test))
#print('Accuracy: %.3f' % scores[1])
