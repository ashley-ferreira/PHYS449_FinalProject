#-----------------------------------------------------
#IMPORT MODULES
from torch import nn
#-----------------------------------------------------

#-----------------------------------------------------
#DEFINE VARIABLES USED IN THE CODE:
num_classes = 4 #Number of classes for the model
num_images = 50 #number of different galaxy images per augmented batch.
#-----------------------------------------------------


networkc1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.ReLU(), 
    nn.Linear(135424,256),
    nn.ReLU(),
    nn.Linear(256, num_classes))