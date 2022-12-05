# run the pip install if you don't have wandb
#!pip install wandb==0.9.7
##wandb login(write a command to input this into terminal)
# if it asks you for a code you can use: 469092e605208488a82954d1b80c92028151663a


from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import wandb
import time

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

num_classes = 4

# need to double check but this is roughly right
networkc1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(32),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(64),

    # max pool here
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),

    #nn.Linear(135424, 256), 
    # dropout here
    nn.Dropout(0.5),
    nn.ReLU(), # do we need an activation function here?
    nn.Linear(135424,256),
    nn.ReLU(),
    nn.Linear(256, num_classes))

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

#LOAD THE DATA FROM TXT FILE INTO A BATCH:
def data_batch(datafile_index, num_images=10, data_file='PHYS449_FinalProject/data/data_g_band.txt', plotting=False):
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

    #Take batch of num_images rows from datafile:
    with open(data_file, 'r') as f:
        rows = f.readlines()[datafile_index:(datafile_index+num_images)]

    #for batch size of 400 (augmented), need 10 images
    data_batch = np.zeros((num_images,12101), dtype=np.dtype('U10'))
    count = 0
    for row in rows:
        data_batch[count,:] = row.split()
        count += 1
        
    input_batch_flat = np.array(data_batch[:,:12100], dtype=int)#, dtype=int)
    label_batch = np.array(data_batch[:,-1])

    #convert input batch back to a 2D array:
    input_batch = np.zeros((110,110,np.shape(input_batch_flat)[0]))#, dtype=int)
    for ii in range(np.shape(input_batch_flat)[0]):
        input_batch[:,:,ii] = np.reshape(input_batch_flat[ii,:], (110,110))


    #convert label batch into into 1D vector: 
    #E=0, S0=1, Sp=2, Irr+Misc=3
    #ex: label = [0,0,1,0] ==> Sp galagy
    arr_label_batch = np.zeros((np.shape(label_batch)[0],4), dtype=int)
    arr_label_batch[:,0] = np.array([label_batch == 'E'], dtype=int)
    arr_label_batch[:,1] = np.array([label_batch == 'Sp'], dtype=int)
    arr_label_batch[:,2] = np.array([label_batch == 'S0'], dtype=int)
    arr_label_batch[:,3] = np.array([label_batch == 'Irr+Misc'], dtype=int)

    if plotting == True:
      #test with image plotted
      plt.imshow(input_batch[:,:,0])
      plt.show()

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

#LESS DATA AUGMENTION: crop only = X5 augmentation:

#AUGMENT ONLY X5 (ONLY BY CROPPING)
def data_batch_aug5(datafile_index, num_images=10,  data_file='PHYS449_FinalProject/data/data_g_band.txt', plotting=False):
    '''
    Description:
        Access datafile.txt, each row is flattened 110x110 image + 1 label string (E, Sp, S0, Irr+Misc).
        Returns an augmented batch of num_images X 5.
        The labels are converted to 1D vectors (ex: Sp = [0,0,1,0])
        Need to give a datafile_index that tells which rows to pick.
    Inputs:
        datafile_index: index of row in datafile to load. loads rows datafile_index to datafile_index+num_images.
        num_images: number of different images to load per batch, total batch size 
        is 5 X num_images. (default: 10 (for 5X10 = 400 batch size like in paper)
        data_file: datafile full path, need to add shortcut to local Drive. (default: '/content/drive/MyDrive/data/data_g_band.txt')
    Outputs:
        tensor_input_batch_aug: dimensions: (100, 100, num_images X 5). 
        tensor_label_batch_aug: dimensions: (num_images X 5, 4)
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
    if plotting == True:
      plt.imshow(input_batch[:,:,0])
      plt.show()

    #NOW AUGMENT THE BATCH (5X more):
    how_much_augment = 5
    input_batch_aug = np.empty((100,100,np.shape(input_batch)[2]*how_much_augment), dtype=int)
    arr_label_batch_aug = np.empty((np.shape(arr_label_batch)[0]*how_much_augment, 4), dtype=int)

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
            input_batch_aug[:,:,count] = C[kk]
            arr_label_batch_aug[count,:] = arr_label_batch[ll,:]
            count += 1

    #PUT THE DATA AS A PYTORCH TENSOR:
    tensor_input_batch_aug = torch.Tensor(input_batch_aug)
    tensor_label_batch_aug = torch.Tensor(arr_label_batch_aug)
    
    return tensor_input_batch_aug, tensor_label_batch_aug

'''
#Test above function:
rand_index = np.random.permutation(1403) #10 images
rand_train = rand_index[:200] #arbitrary values
rand_test = rand_index[200:]

#Use this loop for training over entire dataset at each epochs
for ii in range(np.shape(rand_train)[0]):
  image_batch, label_batch = data_batch_aug5(datafile_index=10*rand_train[ii], num_images=10)
  ##print(np.shape(image_batch))
  ##print(np.shape(label_batch))
  ##print(label_batch)
  #Check: 10 images X 5 augmentation = 100 x 100 x 50 tensor size
  #check: label size is 10 x 5 = 50 x 4 (4 labels)
  #check: label is 5 type in a row then another 5 in a row etc ...
'''

# can just call data load for some if we made plotting an optional arg
rand_index = np.random.permutation(1)#10)
print(np.shape(rand_index)[0])
for ii in range(np.shape(rand_index)[0]):
  image_batch, label_batch = data_batch(datafile_index=50*rand_index[ii], num_images=50, plotting = True)




#Train and test set
num_images = 10
dataset_size = 1403#int(14030/num_images) # WHENEVER YOU SEE THIS LESS THAN 1403 IT'S ARTIFIFICALLY SMALL JUST TO TROUBLESHOOT CODE AND SHOULD NOT BE USED TO TRAIN
train_split = 0.7
test_valid_split = 0.5 # X
test_split = 1 - train_split
split_cutoff = int(dataset_size*train_split)

rand_index = np.random.permutation(dataset_size)
rand_train = rand_index[:split_cutoff] # get these split like paper proportions
rand_test = rand_index[split_cutoff:dataset_size] # valudation will be taken from test set




network_to_train = 'C2'

# define hyperparameters of training
if network_to_train == 'C1':
  n_epochs = 12
  # can't find learning rate mentioned so I'm leaving it as default for now
  cn_model = networkc1
  #optimizer = torch.optim.Adadelta(cn_model.parameters()))
  # trying adam for a sec
  optimizer = torch.optim.Adam(cn_model.parameters(), lr=2e-4)

elif network_to_train == 'C2':
  n_epochs = 20
  cn_model = networkc2
  lr = 2*pow(10,-4)
  optimizer = torch.optim.Adam(cn_model.parameters(), lr=lr)




# connect to w&b for experiment tracking
##wandb.init(project="rough-trails", entity="449-final-project")

##wandb.config = {
##  "learning_rate": lr,
##  "epochs": n_epochs,
##  "model": network_to_train,
##}





# define things that are the same for both notebooks
loss_fn = torch.nn.CrossEntropyLoss() 

# Initialize network & move to GPU
cn_model.to(DEVICE)  # comment out if this gives you issues



# For monitoring acc and losses
avg_epoch_acc_train = []
avg_epoch_acc_val = []
avg_epoch_losses_train = []
avg_epoch_losses_val = []

batch_size = num_images*40#40 

print('Model initialized and prepped, begin training...')
cn_model.train()
for epoch in range(n_epochs):  
    print('epoch:', epoch+1)

    # quick fix to get training dataset size
    ds_size = 0
    
    train_total_accuracy = 0
    epoch_loss = 0

    start_time = time.time()
    for ii in range(np.shape(rand_train)[0]): 
      #print('batch', ii+1, '/', batch_size)
      im, y = data_batch(datafile_index=num_images*rand_train[ii], num_images=num_images)

      # reshaping im to what we want (can do this as data output too)
      im = im.reshape(im.shape[2], 1, 100, 100)

      y_pred = cn_model(im)
      y_pred_cat = nn.functional.softmax(y_pred, dim=1)
      

      #updated accuracy calculation:
      train_predictions = torch.argmax(y_pred, dim=1)
      train_label_predictions = torch.argmax(y, dim=1)
      train_batch_size = np.shape(train_predictions)[0]
      train_batch_accuracy = torch.sum(train_predictions == train_label_predictions).item()/train_batch_size
      print(f'train batch accuracy = {100*train_batch_accuracy} %')
      train_total_accuracy += train_batch_accuracy

      # im doing the backprop after each batch
      # (we may just want to do after each epoch)
      loss = loss_fn(y_pred, y)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      ds_size += 1

    print("--- %s seconds ---" % (time.time() - start_time))

    t_loss = epoch_loss / ds_size
    print('training loss:', t_loss)
    avg_epoch_losses_train.append(t_loss)

    train_total_accuracy = 100 * train_total_accuracy / np.shape(rand_train)[0]
    print('training accuracy:', train_total_accuracy, '%')
    avg_epoch_acc_train.append(train_total_accuracy)

    # Validation
    cn_model.eval()
    epoch_loss = 0
    with torch.no_grad():
      ## quick fix to get some validation data, we want to use more than this 
      for ii in range(np.shape(rand_test)[0]):
        if ii == 0:
          im_valid, y_valid = data_batch(datafile_index=num_images*rand_test[ii], num_images=num_images)
          im_valid = im_valid.reshape(im_valid.shape[2], 1, 100, 100)
        else:
          break
      y_pred_valid = cn_model(im_valid)
      loss = loss_fn(y_pred_valid, y_valid)
      epoch_loss += loss.item()
      v_loss = epoch_loss
      avg_epoch_losses_val.append(v_loss)
      print('validation loss:', v_loss)

      valid_predictions = torch.argmax(y_pred_valid, dim=1)
      valid_label_predictions = torch.argmax(y_valid, dim=1)
      valid_batch_size = np.shape(valid_predictions)[0]
      valid_batch_accuracy = torch.sum(valid_predictions == valid_label_predictions).item()/valid_batch_size
      print(f'Validation accuracy = {100*valid_batch_accuracy} %')
      
      avg_epoch_acc_val.append(valid_batch_accuracy)

      ##wandb.log({"train_loss": t_loss, "valid_loss": v_loss, "train_acc": train_total_accuracy/100, "valid_acc": valid_batch_accuracy}) # these variables were quite right at first
      #wandb.watch(cn_model)

print("DONE TRAINING")

# save model itself 
torch.save(cn_model.state_dict(), 'test_model1')#, CWD + 'Notebooks/models/')


# plot accuracy/loss versus epoch
fig1 = plt.figure(figsize=(10,3))


ax1 = plt.subplot(121)
ax1.plot(avg_epoch_acc_train, '--', color='darkslategray', linewidth=2, label='training')
ax1.plot(avg_epoch_acc_val, linewidth=2, label='valiation') 
ax1.legend()
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')

ax2 = plt.subplot(122)
ax2.plot(avg_epoch_losses_train, '--', color='crimson', linewidth=2, label='training')
ax2.plot(avg_epoch_losses_val, linewidth=2, label='validation')
ax2.legend()
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')

fig1.savefig('PHYS449_FinalProject/Notebooks/plots/'+'CNN_training_history.png')

plt.show()
