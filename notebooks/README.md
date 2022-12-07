# What are all these notebooks?

## General Notes
- all these notebooks have been developed and run in Google Colab
- all notebooks which use the "original data derived from SDSS" require you to have a copy of this dataset in a folder named "data/"
- the notebook whch uses Galaxy10 data will download this automatically for you in the notebook

## CNN_Fully_Augmented_Dataset_Pytorch.ipynb
This notebook contains the code to create the PyTorch 4-way classifier using the C1 and C2 architectures and by training on the original data derived from SDSS

## KerasC1_SDSS_noBatchNorm.ipynb
This folder contains the code for training a 4-way classifier using the C1 archtecture and Keras. This is trained on the original data derived from SDSS. It was not explicitly stated that C1 has batch normalization layers so 

## KerasC1_withBatchNorm.ipynb

## KerasC2_SDSS.ipynb

## KerasGalaxy10_gband.ipynb
This notebook contains the code for training a 3-way and 4-way classifier using both the C1 and C2 architectures on the Galaxy10 data using Keras

## Feature_Maps_Extraction.ipynb
This notebook contains the code to create the feature maps of the galaxies using both the C1 and C2 model architectures
