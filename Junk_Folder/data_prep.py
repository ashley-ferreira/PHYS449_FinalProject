#-----------------------------------------------------------
#DATA PREPARATION AND ACCESS:
#J. Ducatel
#
#The following code describes:
#1. how the data was downloaded
#2. how the relevent parts of the data were selected
#3. how the data was augmented 
#4. how the Nair catalog label converted to Cavanach label
#5. how the label was saved to the correcponding image
#6. how label and input data Pytorch Tensor was saved
#7. HOW TO ACCESS THE AUGMENTED DATA PYTORCH TENSOR FILE
#-----------------------------------------------------------


#-----------------------------------------------------------
#IMPORT MODULES:
import numpy as np
import os, sys #Run the command line input for .sh file
from tqdm import tqdm #add completion bar to code
import imageio as iio #use to save to jpg
import cv2 #use to open jpg
import torch #use to save data in tensor
from torch.utils.data import TensorDataset, DataLoader #use to save data in tensor
#-----------------------------------------------------------


#-----------------------------------------------------------
#DOWNLOAD THE RAW DATA:
'''
#Get RA/DEC from NAIR catalog.
data = np.loadtxt('Nair_2010_Catalog.txt', skiprows=137, usecols=(1,2), dtype=str)

#Download JPG file form SDSS for each RA/DEC
#Scale: 0.7 "/pix found by eye to contain all the galaxy
#size: 110 x 110 pixels
#Use an .sh file for the download command with wget.
for ii in tqdm(range(np.shape(data)[0])):
    command = "sh sdss_cutout_1min_v2.sh " + data[ii,0] + ' ' + data[ii,1] + ' ' + str(ii) + ".jpg"
    os.system(command)

#.sh file source: https://astromsshin.github.io/science/code/index.html 
#the data downloaded is set of 14,034 .jpg files with R,G,B color layers.
'''
#-----------------------------------------------------------


#-----------------------------------------------------------
#CONVERT RGB TO JUST g-band IMAGES:
'''
N = 14034 #number of galaxies in catalogue
#Exctract g-band (B color) and save 3 galaxies per jpg file since 3 layers
for ii in tqdm(range(0,N,3)):
    #Get jpg image into array
    img_layer1 = cv2.imread('Data_download/' + str(ii) +'.jpg')
    img_layer2 = cv2.imread('Data_download/' + str(ii+1) +'.jpg')
    img_layer3 = cv2.imread('Data_download/' + str(ii+2) +'.jpg')

    #Get B value since correspond to g-band in SDSS (source: https://www.sdss.org/dr17/imaging/jpg-images-on-skyserver/)
    img_G = np.empty((110,110,3),dtype=float)
    img_G[:,:,0] = img_layer1[:,:,2]
    img_G[:,:,1] = img_layer2[:,:,2]
    img_G[:,:,2] = img_layer3[:,:,2]

    #Save 3 images 1 jpg with 3 layers to folder
    iio.imwrite('Data_g_band/' + str(ii) + '_to_' + str(ii+2) + '_g_band_norm.jpg', img_G)
'''
#-----------------------------------------------------------


#-----------------------------------------------------------
#AUGMENT THE DATA X40 FROM 14,034 to 561,360:
'''
N = 14034 #number of galaxies in catalogue

#Access Data_g_band data jpg files:
#FILENAME: C = crop to 100X100, R = rotated by 90 deg, M = reflected along x-=y axis
for ii in tqdm(range(0,N,3)):
    #Get jpg image into array
    img_3_layers = cv2.imread('Data_g_band/' + str(ii) + '_to_' + str(ii+2) + '_g_band_norm.jpg')
    #Save 3 layer image with augmentation to augmented folder:
    
    #Crop 5X more image (100X100 pixels)
    C1 = img_3_layers[:100,:100,:]
    C2 = img_3_layers[10:,:100,:]
    C3 = img_3_layers[:100,10:,:]
    C4 = img_3_layers[10:,10:,:]
    C5 = img_3_layers[5:105,5:105,:]
    
    C = [C1, C2, C3, C4, C5]
    
    #Rotate 4X more image (by 90 deg)
    for kk in range(5):
        for jj in range(4):
            C_R = np.rot90(C[kk], k=jj)
            iio.imwrite('Data_g_band_augmented/'+str(ii)+'_to_'+str(ii+2)+'_aug_C'+ 
            str(kk)+'_R'+str(jj)+'_M0.jpg', C_R)
            
            iio.imwrite('Data_g_band_augmented/'+str(ii)+'_to_'+str(ii+2)+'_aug_C'+ 
            str(kk)+'_R'+str(jj)+'_M1.jpg', np.swapaxes(C_R,0,1))
#Data saved in Data_g_band_augmented/ folder
'''
#-----------------------------------------------------------


#-----------------------------------------------------------
#ACCESS DATA LABEL FROM NAIR CATALOGUE & CONVERT TO CAVANACH LABEL:
#Get RA/DEC from NAIR catalog.
Nair_label = np.loadtxt('Nair_2010_Catalog.txt', skiprows=137, usecols=(-12), dtype=int)
#convert NAIR T-Type label to E,S0,Sp,Irr+Misc as described in Cavanach
Cavanach_label = np.empty(np.shape(Nair_label), dtype='U8')

for ii in range(np.shape(Nair_label)[0]):
    if Nair_label[ii] == -5:
        Cavanach_label[ii] = 'E'
    elif -3 <= Nair_label[ii] <= 0:
        Cavanach_label[ii] = 'S0'
    elif 1 <= Nair_label[ii] <= 9:
        Cavanach_label[ii] = 'Sp'
    elif Nair_label[ii] == 10 or Nair_label[ii] == 99:
        Cavanach_label[ii] = 'Irr+Misc'
    else:
        Cavanach_label[ii] = 'Irr+Misc' #this category includes only Nair T-Type label of 11 and 12, which are part of misc but the Cavanagh paper seem to imply there are not values 11 or 12, but 10+ is misc and irr so this labeling makes sence with what Cavanagh did.
#-----------------------------------------------------------


#-----------------------------------------------------------
#ACCESS THE AUGMENTED DATA:
directory = 'Data_g_band_augmented'
count = 0
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    #load the file
    img_3_layers = cv2.imread(f)

    #3 input (g band image) in file:
    input1 = img_3_layers[:,:,0]
    input2 = img_3_layers[:,:,1]
    input3 = img_3_layers[:,:,2]
    
    #get the corresponding 3 labels:
    index_label = int(filename[0])
    
    label1 = Cavanach_label[index_label]
    label2 = Cavanach_label[index_label+1]
    label3 = Cavanach_label[index_label+2]

#CAN NOW LOAD THE DATA AND PROCESS IT
#OPENING AND FINDING THE FILE LABEL FOR ALL FILES TAKES ~ 13 minutes
#WILL SAVE AS A DIFFERENT FORMAT TO ACCESS IN BATCHES WITH PYTORCH LATER  
#-----------------------------------------------------------




