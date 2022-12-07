# What are all these notebooks?

## General Notes
- all these notebooks have been developed and run in Google Colab
- all notebooks which use the "original data derived from SDSS" require you to have a copy of this dataset in a folder named "data/"
- the notebook whch uses Galaxy10 data will download this automatically for you in the notebook

## CNN_Fully_Augmented_Dataset_Pytorch.ipynb
This notebook contains the code to create the PyTorch 4-way classifier using the C1 and C2 architectures and by training on the original data derived from SDSS

## KerasC1_SDSS_noBatchNorm.ipynb
This notebbook contains the code for training a 4-way classifier using the C1 archtecture and Keras. This is trained on the original data derived from SDSS. It was not explicitly stated that C1 has batch normalization layers so this notebook does not impliment any batch normalization.

## KerasC1_SDSS_withBatchNorm.ipynb
This notebook is almost identical to the one above but where batch normalization is used as it was seen to improve the results for the C1 architecture.

## KerasC2_SDSS.ipynb
This notebook follows the previous two, using Keras to impliment a 4-way classifier of the original data derived from SDSS, however this notebook trains the C2 model architecture.

## KerasGalaxy10_gband.ipynb
This notebook contains the code for training a 3-way and 4-way classifier using both the C1 and C2 architectures on the Galaxy10 data using Keras

## Feature_Maps_Extraction.ipynb
This notebook contains the code to create the feature maps of the galaxies using both the C1 and C2 model architectures
