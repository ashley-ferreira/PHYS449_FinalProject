from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.layers.core import Dropout
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


# connect to w&b for experiment tracking
#wandb.init(project="CNN-4way-C1-subset", entity="449-final project")

## Convulational Neural Network "C2"
networkc2 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=3, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=3, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(64),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=3, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(64),

    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=3, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=2),


    nn.Flatten(),
    nn.Linear(1600, 256),
    nn.Dropout(0.5),

    nn.ReLU(),
    nn.Linear(256,4),

    nn.ReLU(),
    nn.Linear(256,4))


## Loading the data into batches

#LOAD THE DATA FROM TXT FILE INTO A BATCH:
def data_batch(datafile_index, num_images=10, data_file='/data/data_g_band.txt'):
    '''
    Description:
        Access datafile.txt, each row is flattened 110x110 image + 1 label string (E, Sp, S0, Irr+Misc).
        Returns an augmented batch of num_images X 40.
        The labels are converted to 1D vectors (ex: Sp = [0,0,1,0])
        Need to give a datafile_index that tells which rows to pick.
    Inputs:
        datafile_index: index of row in datafile to load. loads rows datafile_index to datafile_index+num_images.
        num_images: number of different images to load per batch, total batch size 
        is 40 X num_images. (default: 10 (for 40X10 = 400 batch size like in paper)
        data_file: datafile full path, need to add shortcut to local Drive. (default: '/content/drive/MyDrive/data/data_g_band.txt')
    Outputs:
        tensor_input_batch_aug: dimensions: (100, 100, num_images X 40). 
        tensor_label_batch_aug: dimensions: (num_images X 40, 4)
    '''

    #data_file = 'data_g_band.txt'

    #Take batch of num_images rows from datafile:
    with open(data_file, 'r') as f:
        rows = f.readlines()[datafile_index:(datafile_index+num_images)]

    #for batch size of 400 (augmented), need 10 images
    data_batch = np.zeros((num_images,12101), dtype=np.dtype('U10'))
    count = 0
    for row in rows:
        data_batch[count,:] = row.split()
        count += 1

    #separate label and input:
    input_batch_flat = np.array(data_batch[:,:12100], dtype=int)
    label_batch = np.array(data_batch[:,-1])

    #convert input batch back to a 2D array:
    input_batch = np.empty((110,110,np.shape(input_batch_flat)[0]), dtype=int)
    for ii in range(np.shape(input_batch_flat)[0]):
        input_batch[:,:,ii] = np.reshape(input_batch_flat[ii,:], (110,110))


    #convert label batch into into 1D vector: 
    #E=0, S0=1, Sp=2, Irr+Misc=3
    #ex: label = [0,0,1,0] ==> Sp galagy
    arr_label_batch = np.empty((np.shape(label_batch)[0],4), dtype=int)

    arr_label_batch[:,0] = np.array([label_batch == 'E'], dtype=int)
    arr_label_batch[:,1] = np.array([label_batch == 'Sp'], dtype=int)
    arr_label_batch[:,2] = np.array([label_batch == 'S0'], dtype=int)
    arr_label_batch[:,3] = np.array([label_batch == 'Irr+Misc'], dtype=int)

    #test with image plotted
    #import matplotlib.pyplot as plt
    #plt.imshow(input_batch[:,:,0])
    #plt.show()

    #NOW AUGMENT THE BATCH (40X more):
    input_batch_aug = np.empty((100,100,np.shape(input_batch)[2]*40), dtype=int)
    arr_label_batch_aug = np.empty((np.shape(arr_label_batch)[0]*40, 4), dtype=int)

    count = 0
    for ll in range(np.shape(input_batch)[2]):
        #Crop 5X more image (100X100 pixels)
        C1 = input_batch[:100,:100,ll]
        C2 = input_batch[10:,:100,ll]
        C3 = input_batch[:100,10:,ll]
        C4 = input_batch[10:,10:,ll]
        C5 = input_batch[5:105,5:105,ll]

        C = [C1, C2, C3, C4, C5]

        for kk in range(5):
            #Rotate 4X more image (by 90 deg)
            for jj in range(4):
                C_R = np.rot90(C[kk], k=jj)
                input_batch_aug[:,:,count] = C_R
                arr_label_batch_aug[count,:] = arr_label_batch[ll,:]
                count += 1
                
                input_batch_aug[:,:,count] = np.swapaxes(C_R,0,1)
                arr_label_batch_aug[count,:] = arr_label_batch[ll,:]
                count += 1


    #PUT THE DATA AS A PYTORCH TENSOR:
    tensor_input_batch_aug = torch.Tensor(input_batch_aug)
    tensor_label_batch_aug = torch.Tensor(arr_label_batch_aug)
    
    return tensor_input_batch_aug, tensor_label_batch_aug


##splitting up the data

#Train and test set:
rand_index = np.random.permutation(280)

# MAKING AN ARTIFICIALLY SMALL DATASET SET FOR NOW
rand_train = rand_index[:10]#[:200] #  does this do 200*50 for number of examples?
rand_test = rand_index[10:20]#[200:] # valudation will be taken from test set


network_to_train = 'C2'

if network_to_train == 'C1':
  n_epochs = 13
  # can't find learning rate mentioned so I'm leaving it as default for now
  cn_model = networkc1
  optimizer = torch.optim.Adadelta(cn_model.parameters())#, lr=2e-5)

elif network_to_train == 'C2':
  n_epochs = 20
  cn_model = networkc2
  optimizer = torch.optim.Adam(cn_model.parameters())#, lr=2e-5)

batch_size = np.shape(rand_train)[0]
valid_split = 0.1

print('Model initialized and prepped, begin training...')
cn_model.train()
for epoch in range(n_epochs):  # progress bar
    print('epoch:', epoch+1)

    # quick fix to get dataset size
    ds_size = 0

    epoch_loss = 0
    for ii in range(batch_size): 
      print('batch', ii+1, '/', batch_size)
      im, y = data_batch(datafile_index=50*rand_train[ii], num_images=batch_size)

      # reshaping im to what we want (can do this as data output too)
      im = im.reshape(im.shape[2], 1, 100, 100)

      y_pred = cn_model(im)
      loss = loss_fn(y_pred, y)
      #print(loss)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      ds_size += 1

    t_loss = epoch_loss / ds_size
    print('training loss:', t_loss)
    avg_epoch_losses_train.append(t_loss)

    t_acc = torch.sum(y_pred == y).numpy()/(4*ds_size)
    print('training accuracy:', t_acc)
    avg_epoch_acc_train.append(t_acc)

    # Validation
    cn_model.eval()
    epoch_loss = 0
    with torch.no_grad():
      # add unaugmented stuff for test and maybe validation? below is a quick fix
      for ii in range(np.shape(rand_test)[0]):
        if ii == 0:
          im_valid, y_valid = data_batch(datafile_index=50*rand_test[ii], num_images=50)
          im_valid = im_valid.reshape(im_valid.shape[2], 1, 100, 100)
        else:
          pass
      y_pred_valid = cn_model(im_valid)
      loss = loss_fn(y_pred_valid, y_valid)
      epoch_loss += loss.item()
      v_loss = epoch_loss
      avg_epoch_losses_val.append(v_loss)
      print('validation loss:', v_loss)

      v_acc = torch.sum(y_pred_valid == y_valid).numpy()/(4*ds_size)
      print('validation accuracy:', v_acc)
      vg_epoch_acc_train.append(v_acc)

      #wandb.log({"loss": loss, "validation_loss": v_loss})
      #wandb.watch(cn_model)

# to do: add in C2, add in accuracy calculations and tracking, change variable names and all to get plotting below working

