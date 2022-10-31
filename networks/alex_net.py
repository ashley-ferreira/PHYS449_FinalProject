from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.layers.core import Dropout

def Alex(input_shape, unique_labels=2, dropout_rate=0.5):
    '''
    Defines the 2D Convolutional Neural Network (CNN)
    Parameters:    
        input_shape (arr): input shape for network
        training_labels (arr): training labels
        unique_labels (int): number unique labels 
        dropout_rate (float): dropout rate as fraction
    Returns:
        
        model (keras model class): convolutional neural network to train
    '''

    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=input_shape, activation='relu', padding='same', kernel_size=(11,11)))
    model.add(MaxPool2D(pool_size=(3,3), padding='valid'))

    model.add(Conv2D(filters=256, input_shape=input_shape, activation='relu', padding='same', kernel_size=(5,5)))
    model.add(MaxPool2D(pool_size=(3,3), padding='valid'))

    model.add(Conv2D(filters=384, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(Conv2D(filters=384, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(Conv2D(filters=256, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(MaxPool2D(pool_size=(3,3), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(unique_labels, activation='softmax')) 

    return model