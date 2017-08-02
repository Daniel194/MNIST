import numpy as np
import pandas as pd
import matplotlib
import keras
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
matplotlib.use('Agg')

epochs = 25
train_size = 42000
test_size = 2000
batch_size = 128
num_classes = 10

img_rows = 28
img_cols = 28

data = pd.read_csv("input/train.csv", nrows=42000)
(train, test) = (data[:train_size], data[42000 - test_size:42000])

x_train = (train.ix[:, 1:].values).astype('float32')
y_train = train.ix[:, 0].values.astype('int32')

x_test = (test.ix[:, 1:].values).astype('float32')
y_test = (test.ix[:, 0].values).astype('int32')

x_test_raw = x_test

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)
y_test_nv = y_test

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("Baseline Error: %.2f%%" % (100 - score[1] * 100))

pred = model.predict_classes(x_test, verbose=0)


def plot_failures(pred, x_test, y_test_nv, file):
    cmp = np.column_stack((pred, y_test_nv))
    df = pd.DataFrame(cmp)
    df['match'] = np.where(df[0] == df[1], 1, 0)
    fail = df[df['match'] == 0].index.tolist()
    x_fail = x_test[np.array(fail)]

    if (len(x_fail) > 100):
        x_fail = x_fail[:100]

    num_images, img = x_fail.shape
    x_fail = x_fail.reshape(num_images, 28, 28)
    x_fail = np.concatenate((x_fail, np.zeros((100 - num_images, 28, 28))), axis=0)

    fig = plt.figure()
    for x in range(1, 10):
        for y in range(1, 10):
            ax = fig.add_subplot(10, 10, 10 * y + x)
            ax.matshow(x_fail[10 * y + x], cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.savefig(file + '.png')


plot_failures(pred, x_test_raw, y_test_nv, "failures")

comp = pd.read_csv("input/test.csv")
x_comp = comp.iloc[:, :].values
x_comp = x_comp.reshape(x_comp.shape[0], 1, img_rows, img_cols)

pred = model.predict_classes(x_comp, verbose=0)

submissions = pd.DataFrame({"ImageId": list(range(1, len(pred) + 1)), "Label": pred})
submissions.to_csv("output/submission1.csv", index=False, header=True)
