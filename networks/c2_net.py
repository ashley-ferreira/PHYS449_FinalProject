from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.layers.core import Dropout

def C2(input_shape, unique_labels=2, dropout_rate=0.5):
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

    model.add(Conv2D(filters=32, input_shape=input_shape, activation='relu', padding='same', kernel_size=(7,7)))
    model.add(MaxPool2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(filters=64, input_shape=input_shape, activation='relu', padding='same', kernel_size=(5,5)))
    model.add(Conv2D(filters=64, input_shape=input_shape, activation='relu', padding='same', kernel_size=(5,5)))
    model.add(MaxPool2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(filters=128, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(MaxPool2D(pool_size=(2,2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(256, activation='sigmoid'))

    model.add(Dense(unique_labels, activation='softmax')) 

    return model