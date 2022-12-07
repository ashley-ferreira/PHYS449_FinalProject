#-----------------------------------------------------
#Pytorch C2
#Dec. 6, 2022
#Train Pytorch Model for C2 Architecture on
#4 way classification for fully augmented data.
#Note: Requires High Ammount of GPU
#-----------------------------------------------------

#-----------------------------------------------------
#IMPORT MODULES
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
#-----------------------------------------------------

## Call these to ensure PyTorch_Data_Loading and PyTorch_C1_net files can be called
sys.path.insert(0, 'Data')
sys.path.insert(1, 'Networks')

#-----------------------------------------------------
#IMPORT DEFINED FUNCTIONS:
from PyTorch_Data_Loading import data_batch


from PyTorch_C2_net import networkc2
#-----------------------------------------------------


num_classes = 4 #Number of classes for the model
num_images = 150 #number of different galaxy images per augmented batch.
n_epochs = 20
cn_model = networkc2
lr = 2*pow(10,-4)
optimizer = torch.optim.Adam(cn_model.parameters(), lr=lr)

loss_fn = torch.nn.CrossEntropyLoss()

cn_model.to('cpu') #Change to 'cpu' to 'cuda' if running network on a GPU
#-----------------------------------------------------


#-----------------------------------------------------
#DEFINE TRAINING AND TESTING SETS:
#Train and test set
dataset_size = int(14034/num_images)
train_split = 0.85
test_split = 1 - train_split
split_cutoff = int(dataset_size*train_split)

rand_index = np.random.permutation(dataset_size)
rand_train = rand_index[:split_cutoff]
rand_test = rand_index[split_cutoff:dataset_size]
#-----------------------------------------------------


#-----------------------------------------------------
#TRAIN THE MODEL:
# For monitoring acc and losses
avg_epoch_acc_train = []
avg_epoch_acc_val = []
avg_epoch_losses_train = []
avg_epoch_losses_val = []

batch_size = num_images*40 #40 is the 40X augmentation

print('Model initialized and prepped, begin training...')

for epoch in range(n_epochs):  
    cn_model.train()
    print('epoch:', epoch+1)

    #VALIDATION FOR before any training! (just once)
    if epoch == 0:
        ds_valid_size = 0
        cn_model.eval() #dont know what that does
        epoch_loss = 0
        test_total_accuracy = 0
        with torch.no_grad():
          for ii in range(np.shape(rand_test)[0]):
            im_valid, y_valid = data_batch(datafile_index=num_images*rand_test[ii], num_images=num_images)
            im_valid = im_valid.reshape(100, 100, 1, im_valid.shape[2])
            im_valid = im_valid.T

            #print(np.shape(im_valid))
            #plt.imshow(im_valid[0,0,:,:])
            #plt.show()

            im_valid = im_valid.detach().to('cpu') #Change to 'cpu' to 'cuda' if running network on a GPU
            y_valid = y_valid.detach().to('cpu') #Change to 'cpu' to 'cuda' if running network on a GPU

            y_pred_valid = cn_model(im_valid)
            y_pred_valid_cat = nn.functional.softmax(y_pred_valid, dim=1)

            #updated accuracy calculation:
            test_predictions = torch.argmax(y_pred_valid_cat, dim=1)
            #test_predictions = torch.argmax(y_pred_valid, dim=1)
            test_label_predictions = torch.argmax(y_valid, dim=1)
            test_batch_size = np.shape(test_predictions)[0]
            test_batch_accuracy = torch.sum(test_predictions == test_label_predictions).item()/test_batch_size
            print(f'\t\t test batch accuracy = {np.round(100*test_batch_accuracy,2)} %, batch # {ds_valid_size}')
            test_total_accuracy += test_batch_accuracy

            loss = loss_fn(y_pred_valid, y_valid)
            epoch_loss += loss.item()
            ds_valid_size += 1

            #delete image and label every loop train:
            del im_valid
            del y_valid
            torch.cuda.empty_cache()
          
          #calculate total loss validation
          v_loss = epoch_loss / ds_valid_size
          avg_epoch_losses_val.append(v_loss)
          print('validation loss:', np.round(v_loss,2))

          #calculate total accuracy validation
          test_total_accuracy = 100 * test_total_accuracy / np.shape(rand_test)[0]
          print('Validation accuracy:', np.round(test_total_accuracy,2), '%')
          avg_epoch_acc_val.append(test_total_accuracy)



    # quick fix to get training dataset size
    ds_size = 0
    
    train_total_accuracy = 0
    epoch_loss = 0
    for ii in range(np.shape(rand_train)[0]):
      optimizer.zero_grad() #reset the gradients (Added)

      #print('batch', ii+1, '/', batch_size)
      im2, y = data_batch(datafile_index=num_images*rand_train[ii], num_images=num_images)

      # reshaping im to what we want (can do this as data output too)
      im2 = im2.reshape(100, 100, 1, im2.shape[2])
      im = im2.T

      del im2
      #torch.cuda.empty_cache()

      im = im.detach().to('cpu') #Change to 'cpu' to 'cuda' if running network on a GPU
      y = y.detach().to('cpu') #Change to 'cpu' to 'cuda' if running network on a GPU

      y_pred = cn_model(im)
      y_pred_cat = nn.functional.softmax(y_pred, dim=1)
      #print(np.shape(y_pred))
      #print(np.shape(y_pred_cat))
      

      #updated accuracy calculation:
      #train_predictions = torch.argmax(y_pred, dim=1)
      train_predictions = torch.argmax(y_pred_cat, dim=1)
      train_label_predictions = torch.argmax(y, dim=1)
      train_batch_size = np.shape(train_predictions)[0]
      train_batch_accuracy = torch.sum(train_predictions == train_label_predictions).item()/train_batch_size
      print(f'\t\t train batch accuracy = {np.round(100*train_batch_accuracy,2)} %, batch # {ds_size}')
      train_total_accuracy += train_batch_accuracy

      # im doing the backprop after each batch
      # (we may just want to do after each epoch)
      #loss = loss_fn(y_pred_cat, y)
      loss = loss_fn(y_pred, y) #commented out because this does not have the softmax applied to it (does it?)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      ds_size += 1

      #delete image and label every loop train:
      #im = im.detach() #try to make sure not saved image and label
      #y = y.detach()

      del im
      del y
      torch.cuda.empty_cache()

    t_loss = epoch_loss / ds_size
    print('training loss:', np.round(t_loss,2))
    avg_epoch_losses_train.append(t_loss)

    train_total_accuracy = 100 * train_total_accuracy / np.shape(rand_train)[0]
    print('training accuracy:', np.round(train_total_accuracy,2), '%')
    avg_epoch_acc_train.append(train_total_accuracy)


    #NEW VALIDATION:----------------------------------

    ds_valid_size = 0
    cn_model.eval() #what does that do?
    epoch_loss = 0
    test_total_accuracy = 0
    with torch.no_grad():
      for ii in range(np.shape(rand_test)[0]):
        im_valid, y_valid = data_batch(datafile_index=num_images*rand_test[ii], num_images=num_images)
        im_valid = im_valid.reshape(100, 100, 1, im_valid.shape[2])
        im_valid = im_valid.T

        im_valid = im_valid.detach().to('cpu') #Change to 'cpu' to 'cuda' if running network on a GPU
        y_valid = y_valid.detach().to('cpu') #Change to 'cpu' to 'cuda' if running network on a GPU

        y_pred_valid = cn_model(im_valid)
        y_pred_valid_cat = nn.functional.softmax(y_pred_valid, dim=1)

        #updated accuracy calculation:
        test_predictions = torch.argmax(y_pred_valid_cat, dim=1)
        #test_predictions = torch.argmax(y_pred_valid, dim=1)
        test_label_predictions = torch.argmax(y_valid, dim=1)
        test_batch_size = np.shape(test_predictions)[0]
        test_batch_accuracy = torch.sum(test_predictions == test_label_predictions).item()/test_batch_size
        print(f'\t\t test batch accuracy = {np.round(100*test_batch_accuracy,2)} %, batch # {ds_valid_size}')
        test_total_accuracy += test_batch_accuracy

        loss = loss_fn(y_pred_valid, y_valid)
        epoch_loss += loss.item()
        ds_valid_size += 1

        #delete image and label every loop train:
        del im_valid
        del y_valid
        torch.cuda.empty_cache()
      
      #calculate total loss validation
      v_loss = epoch_loss / ds_valid_size
      avg_epoch_losses_val.append(v_loss)
      print('validation loss:', np.round(v_loss,2))

      #calculate total accuracy validation
      test_total_accuracy = 100 * test_total_accuracy / np.shape(rand_test)[0]
      print('Validation accuracy:', np.round(test_total_accuracy,2), '%')
      avg_epoch_acc_val.append(test_total_accuracy)


print("DONE TRAINING")
#-----------------------------------------------------


#-----------------------------------------------------
#SAVE PLOTS AND MODEL TO RESULTS Folder:

train_acc = np.array(avg_epoch_acc_train)
valid_acc = np.array(avg_epoch_acc_val)
train_loss = np.array(avg_epoch_losses_train)
valid_loss = np.array(avg_epoch_losses_val)

#Plot accuracy results:
plt.figure(figsize=(7,5)) #set plot size

plt.plot(range(np.shape(train_acc)[0]), train_acc, label='Training Accuracy', 
             linestyle='-', color='red', linewidth=3)
plt.plot(range(np.shape(valid_acc)[0]), valid_acc, label='Validation Accuracy', 
             linestyle='-', color='blue', linewidth=3)

plt.yticks(fontsize=12, rotation=0) #adjust axis tick numbers font size
plt.xticks(fontsize=12, rotation=0) #adjust axis tick numbers font size
plt.xlabel('Epoch Number', fontsize=14) #set axis label
plt.ylabel('Percent Accuracy', fontsize=14) #set axis label
plt.title('Training of 4 way C2 CNN Network', fontsize=16) #set title
plt.legend(fontsize=10)
plt.xlim(0, np.shape(train_acc)[0]-1) #set axis limits
plt.grid(True, which='minor', color='gray', linestyle='--', linewidth=1, alpha=0.2) #set gridlines
plt.grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.5) #set gridlines
plt.tight_layout()
plt.savefig('PHYS449_FinalProject/results/plots/C2_4way_Full_Augmentation_Accuracy_plot.png',dpi=300)
plt.close() #Stops the figure from being shown
#plt.show() #display the figure



#Plot loss results:
plt.figure(figsize=(7,5)) #set plot size

plt.plot(range(np.shape(train_loss)[0]), train_loss, label='Training Loss', 
             linestyle='-', color='red', linewidth=3)
plt.plot(range(np.shape(valid_loss)[0]), valid_loss, label='Validation Loss', 
             linestyle='-', color='blue', linewidth=3)

plt.yticks(fontsize=12, rotation=0) #adjust axis tick numbers font size
plt.xticks(fontsize=12, rotation=0) #adjust axis tick numbers font size
plt.xlabel('Epoch Number', fontsize=14) #set axis label
plt.ylabel('Loss', fontsize=14) #set axis label
plt.title('Training of 4 way C2 CNN Network', fontsize=16) #set title
plt.legend(fontsize=10)
plt.xlim(0, np.shape(train_acc)[0]-1) #set axis limits
plt.grid(True, which='minor', color='gray', linestyle='--', linewidth=1, alpha=0.2) #set gridlines
plt.grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.5) #set gridlines
plt.tight_layout()
#plt.yscale('log')
plt.savefig('PHYS449_FinalProject/results/plots/C2_4way_Full_Augmentation_Loss_plot.png',dpi=300)
plt.close() #Stops the figure from being shown
#plt.show() #display the figure


#Save the pytorch model:
torch.save(cn_model.state_dict(), 'PHYS449_FinalProject/results/models/C2_4way_Full_Augmentation_model')

#-----------------------------------------------------
