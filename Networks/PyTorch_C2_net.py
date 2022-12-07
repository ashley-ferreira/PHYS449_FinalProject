from torch import nn

#-----------------------------------------------------
#DEFINE VARIABLES USED IN THE CODE:
num_classes = 4 #Number of classes for the model
num_images = 150 #number of different galaxy images per augmented batch.
#-----------------------------------------------------

networkc2 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(64),

    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(8192, 256),
    nn.Dropout(0.5),

    nn.ReLU(),
    nn.Linear(256,256),
    nn.ReLU(),
    nn.Linear(256, num_classes))