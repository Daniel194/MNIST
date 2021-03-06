import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
import keras
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
from keras.models import load_model

k.set_image_dim_ordering('th')
matplotlib.use('Agg')

FULL_PATH = '/home/ldaniel/Desktop/MNIST/CNN/'

class MNIST(object):
    def __init__(self):
        self.epochs = 25
        self.train_size = 42000
        self.test_size = 2000
        self.batch_size = 128
        self.num_classes = 10

        self.img_rows = 28
        self.img_cols = 28

        self.model = Sequential()

    def predict_classes(self, images, load_mode=False):
        if load_mode:
            self.model = load_model(FULL_PATH + 'checkpoint/model.h5')

        return self.model.predict_classes(images, verbose=0)

    def predict(self, image):
        self.model = load_model(FULL_PATH + 'checkpoint/model.h5')
        return self.model.predict(image, verbose=0)

    def train(self):
        self.__read_data()

        self.__build()

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(self.x_test, self.y_test))

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        print("Baseline Error: %.2f%%" % (100 - score[1] * 100))

        self.model.save(FULL_PATH + 'checkpoint/model.h5')

    def __read_data(self):
        data = pd.read_csv(FULL_PATH + "input/train.csv", nrows=42000)
        (train, test) = (data[:self.train_size], data[42000 - self.test_size:42000])

        x_train = train.ix[:, 1:].values.astype('float32')
        y_train = train.ix[:, 0].values.astype('int32')

        x_test = test.ix[:, 1:].values.astype('float32')
        y_test = test.ix[:, 0].values.astype('int32')

        x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
        self.input_shape = (1, self.img_rows, self.img_cols)

        x_train /= 255
        x_test /= 255

        y_train = np_utils.to_categorical(y_train, self.num_classes)
        y_test = np_utils.to_categorical(y_test, self.num_classes)

        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test = y_test

    def __build(self):
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))


if __name__ == '__main__':

    model = MNIST()

    if sys.argv[1] == 'train':
        model.train()
    elif sys.argv[1] == 'predict':
        comp = pd.read_csv(FULL_PATH + "input/test.csv")
        x_comp = comp.iloc[:, :].values.astype('float32')
        x_comp = x_comp.reshape(x_comp.shape[0], 1, 28, 28)
        x_comp /= 255

        pred = model.predict_classes(x_comp, load_mode=True)

        submissions = pd.DataFrame({"ImageId": list(range(1, len(pred) + 1)), "Label": pred})
        submissions.to_csv(FULL_PATH + "output/submission.csv", index=False, header=True)
    elif sys.argv[1] == 'image_prediction':
        img = img = Image.open('/home/ldaniel/Desktop/MNIST/backend/src/main/resources/image/image.png')
        img.thumbnail((28, 28), Image.ANTIALIAS)

        pix = np.array(img)
        pix = pix[:, :, 3]

        pred = model.predict(pix.reshape(1, 1, 28, 28))

        submissions = pd.DataFrame({"Nr": list(range(0, len(pred[0, :]))), "Acc": pred[0, :]})
        submissions.to_csv(FULL_PATH + "output/image_prediction.csv", index=False, header=True)
