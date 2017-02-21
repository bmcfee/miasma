# CREATED: 2/20/17 20:10 by Justin Salamon <justin.salamon@nyu.edu>

import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history):
    '''
    Plot training curves from keras history object.

    Parameters
    ----------
    history

    Returns
    -------

    '''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

