import os
import sys
import glob
import argparse

import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from keras.utils import plot_model


# argparse
parser = argparse.ArgumentParser(description='Fit or predict')
parser.add_argument('action', choices=['fit', 'predict', 'visualize'])
parser.add_argument('--image-path', '-i', dest='image_path', action='store',
                    type=str, default='', help='image file path')
parser.add_argument('--epochs', '-e', dest='epochs', action='store',
                    type=int, default=-1, help='image file path')
args = parser.parse_args()


X = []
Y = []
model_weight_file = 'flower.weights'
model_file = 'flower.model'
image_dir='./images/'


def prepare():
    global model_file
    global X
    global Y
    model = None

    if os.path.exists(model_file):
        print("loading the model...")
        model = keras.models.load_model(model_file)
        print("model loaded")
        return (model, None, None, None, None)

    index = 0

    for directory in directories:
        img_pattern = os.path.join(image_dir, directory+'/*.jpg')
        print("reading {}:{}...".format(index, directory))
        for img_path in glob.glob(img_pattern):
            img = load_img(img_path, target_size=(64, 64))
            x = img_to_array(img)
            X.append(x)
            Y.append(index)
        index += 1

    X = np.asarray(X)
    Y = np.asarray(Y)

    # change pixel -> 0..1 float value
    X = X.astype('float32')
    X = X / 255.0

    # change category
    Y = np_utils.to_categorical(Y, len(directories))

    # split it for learning (x_*) and test (y_*)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=111)

    if model == None:
        model = create_model(x_train, x_test, y_train, y_test)
    return (model, x_train, x_test, y_train, y_test)


def usage():
    print('flower.py fit --epochs 100')
    print('flower.py visualize # model.png generated')
    print('flower.py predict --image-path foo/bar.jpeg')


def main():
    global model_file
    global X
    global Y
    global directories

    directories = [x for x in next(os.walk('images'))][1]

    index = 0

    if args.action == 'fit' and args.epochs < 0:
        usage()
        return
    if args.action == 'predict' and args.image_path == '':
        usage()
        return

    print('making or loading a model...')
    (model, x_train, x_test, y_train, y_test) = prepare()
    print('model created/loaded')

    if args.action == 'visualize':
        plot_model(model, to_file='model.png')
        return

    if os.path.exists(model_weight_file):
        model.load_weights(model_weight_file)

    if args.action == "fit":
        fit(model, x_train, x_test, y_train, y_test, args.epochs)
        predict_test(model, x_train, x_test, y_train, y_test)

    if args.action == "predict":
        result = predict(model, args.image_path)


def create_model(x_train, x_test, y_train, y_test):
    # make a CNN
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
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
    model.add(Dense(len(directories)))       # number of flowers
    model.add(Activation('softmax'))
    model.summary()

    # compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])
    model.save(model_file)
    return model


def fit(model, x_train, x_test, y_train, y_test, epochs):
    # execute
    history = model.fit(x_train, y_train, epochs=epochs, steps_per_epoch=10)

    # save
    model.save_weights(model_weight_file)

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(['acc', 'val_acc'], loc='lower right')
    # plt.show()


def predict_test(model, x_train, x_test, y_train, y_test):
    # apply test data
    predict_classes = model.predict_classes(x_test)

    # merge
    mg_df = pd.DataFrame(
        {'predict': predict_classes, 'class': np.argmax(y_test, axis=1)})

    # confusion matrix
    ct = pd.crosstab(mg_df['class'], mg_df['predict'])
    print("crolltab: {}".format(ct))


def predict(model, img_path):

    img = load_img(img_path, target_size=(64, 64))
    x = img_to_array(img)
    X = []
    X.append(x)
    X = np.asarray(X)
    X = X.astype('float32')
    X = X / 255.0

    result = model.predict(X)
    print("result: {}".format(result))
    idx = np.argmax(result)
    print("predicted: {}, {}%".format(directories[idx], result[0][idx]*100))

    return result


if __name__ == '__main__':
    main()
