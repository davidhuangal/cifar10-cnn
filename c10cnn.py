import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation
from keras.layers import Dense, Dropout, Flatten


### DATA SETUP

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /=255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

width, height, depth = X_train[0].shape
num_classes = 10

##############

### Convolutional Network

# Architecture:
# INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(width, height, depth)))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
MaxPool2D(pool_size=(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
MaxPool2D(pool_size=(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
MaxPool2D(pool_size=(2, 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

### Train the Model
model.fit(X_train, y_train, epochs=30, batch_size=128)

########################
### Evaluating the model
loss, acc = model.evaluate(X_test, y_test)

print("Accuracy: {}".format(acc))
print("Loss: {}".format(loss))
