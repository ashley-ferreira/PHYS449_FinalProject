#-----------------------------------------------------------
#DATA PREPARATION AND ACCESS:
#
#The following code describes:
#1. how the data was downloaded
#2. how the relevent parts of the data were selected
#4. how the Nair catalog label converted to Cavanach label
#5. how the label was saved to the correcponding image
#NOTE: Only run to download all SDSS images (14034 files)
#-----------------------------------------------------------


#-----------------------------------------------------------
#IMPORT MODULES:
print('IMPORT MODULES', end='')
import numpy as np
import os, sys #Run the command line input for .sh file
from tqdm import tqdm #add completion bar to code
import imageio as iio #use to save to jpg
import cv2 #use to open jpg
import csv #save data to csv file
print(' ==========> DONE')
#-----------------------------------------------------------


#-----------------------------------------------------------
#DOWNLOAD THE RAW DATA FROM SDSS:
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
#Save to downloed data folder each jpg images.
'''
#-----------------------------------------------------------


#-----------------------------------------------------------
#GET THE NAIR LABELS:
'''
print("GET NAIR LABELS", end=' ')
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
print("----------> DONE")
'''


#-----------------------------------------------------------
#SAVE data to txt file:
'''
#Write output result in file_name.out file
output_file_name = "data_g_band_v2.txt"
output_file_out = open(output_file_name, "w") #create the file

directory = 'Data/Data_download'
count = 0

for ii in tqdm(range(14034)):
    filename = str(ii)+'.jpg'
    f = os.path.join(directory, filename)
    #load the file
    img_3_layers = cv2.imread(f)

    #3 input (g band image) in file (not normalized):
    img_G = img_3_layers[:,:,2]
    
    
    label1 = Cavanach_label[ii]
    
    #flatten data array to fit into CSV file line:
    input1_flat = np.ndarray.flatten(img_G)
    
    #append the label at the end of the flattened input:
    input1_csv = np.append(input1_flat, label1)
    
    #write to txt file rows:
    for kk in range(np.shape(input1_csv)[0]-1):
        print(int(input1_csv[kk]), end=' ', file=output_file_out)
    print(input1_csv[-1], file=output_file_out)
    
    
    #plt.figure(figsize=(10,10)) #set plot size
    #plt.imshow(img_G)
    #plt.show()
    
    
    #keep track and stop once enough in datafile
    count += 1
    if count >= 14034:
        break

#close the file
output_file_out.close()
'''
#-----------------------------------------------------------


#-----------------------------------------------------------
#SAVE 3WAY data file: (Used for 3 way classification)

#get location in file of irregular galaxies:
'''
label = []
count = 0
data_file = 'data_g_band.txt'
with open(data_file, 'r') as f:
  while True:
    line = f.readline()
    #split the line    
    data_batch = line.split()
    #get label:
    label.append(data_batch[-1])
    count += 1
    percent = 100*count/14034
    if count % 2000 == 0:
      print(f'{np.round(percent,2)} % read')
    
    if count == 14034:
      break

irr_index = np.where(np.array(label, dtype=str) == 'Irr+Misc')[0]

#print(irr_index[:10])

#get indexes of new file that want to copy:
index_all = np.array(range(14034))
concat = np.concatenate((index_all, irr_index))
unique = np.unique(concat, return_counts=True)
check_double = np.where(unique[1] == 1)
index_3way = unique[0][check_double]

#print(np.shape(index_3way))
'''

#now save to new file:
#Write output result in file_name.out file
'''
output_file_name = "data_g_band_3way.txt"
output_file_out = open(output_file_name, "w") #create the file

count = 0
with open(data_file, 'r') as f:
  while True:
    line = f.readline() #read the count line
    
    #check if line number in index_3way:
    if count in index_3way:
        print(line, end='', file=output_file_out)
           
    count += 1
    #keep trackof progress
    percent = 100*count/14034
    if count % 2000 == 0:
      print(f'{np.round(percent,2)} % write')
    
    if count == 14034:
      break


#close the file
output_file_out.close()
'''

#-----------------------------------------------------------




