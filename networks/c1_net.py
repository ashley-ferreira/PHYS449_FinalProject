from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.layers.core import Dropout

def C1(input_shape, unique_labels=2, dropout_rate=0.5):
    '''
    Defines the 2D Convolutional Neural Network (CNN)
    Parameters:    
    
        input_shape (arr): input shape for network
        unique_labels (int): number unique labels 
        dropout_rate (float): dropout rate as fraction

    Returns:
        
        model (keras model class): convolutional neural network to train
    '''

    model = Sequential()

    model.add(Conv2D(filters=32, input_shape=input_shape, activation='relu', kernel_size=(5,5)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, input_shape=input_shape, activation='relu', kernel_size=(5,5)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation='relu'))

    model.add(Dense(unique_labels, activation='softmax')) 

    return model