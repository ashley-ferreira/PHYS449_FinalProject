# What are all these notebooks?

## General Notes
- all these notebooks have been developed and run in Google Colab
- all notebooks which use the "original data derived from SDSS" require you to have a copy of this dataset in a folder named "data/"
- the notebook whch uses Galaxy10 data will download this automatically for you in the notebook

## CNN_Fully_Augmented_Dataset_Pytorch.ipynb
This notebook contains the code to create the PyTorch 4-way classifier using the C1 and C2 architectures and by training on the original data derived from SDSS

## 
This folder also contains 3-way and 4-way classification was done using keras for the original data derived from SDSS.

## KerasGalaxy10_gband.ipynb
This notebook contains the code for training a 3-way and 4-way classifier using both the C1 and C2 architectures on the Galaxy10 data using Keras

## Feature_Maps_Extraction.ipynb
This notebook contains the code to create the feature maps of the galaxies using both the C1 and C2 model architectures
