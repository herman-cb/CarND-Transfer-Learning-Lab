import pickle
import tensorflow as tf
import numpy as np
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
# TODO: import Keras layers you need here

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    nb_classes = len(np.unique(y_train))
    model = Sequential()
    model.add(Convolution2D(512, 1, 1, border_mode='valid',input_shape=(1, 1, 512)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # TODO: train your model here

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_one_hot = lb.transform(y_train)
    a = -0.5
    b = 0.5
    _min = np.min(X_train)
    _max = np.max(X_train)
    X_train_normalized = a + (b-a) * (X_train - _min)/ (_max - _min)
    history = model.fit(X_train_normalized, y_train_one_hot, batch_size=128, nb_epoch=512, validation_split=0.2)

    X_val_normalized = a + (b-a) * (X_val - _min)/ (_max - _min)
    y_val_one_hot = lb.transform(y_val)
    print('Testing the model.')
    print('{}'.format(model.metrics_names))
    print('{}'.format(model.evaluate(X_val_normalized, y_val_one_hot)))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
