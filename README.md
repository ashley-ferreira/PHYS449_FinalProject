# PHYS 449 Final Project: Galaxy Morphology Classification with Machine Learning

## Contributors
- Yusuf Ahmed (UWaterloo)
- Alexander Caires (UWaterloo)
- Jordan Ducatel (UWaterloo)
- Ashley Ferreira (UWaterloo)
- Guillaume Hewitt (UWaterloo)

## Introduction

Reproducing results from the 2D Convolutional Neural Networks C1 and C2 from "Morphological classification of galaxies with deep learning: comparing 3-way and 4-way CNNs" by Mitchell K. Cavanagh, Kenji Bekki and Brent A. Groves (https://academic.oup.com/mnras/article/506/1/659/6291200) and exploring their application to the Galaxy10 dataset.

tried to replicate the results from the above paper for their "C2" network and their old "C1" network using PyTorch and Keras. In the paper they had originally used Keras to run their neural networks. We decided to run the neural networks using PyTorch and Keras to compare the modules.

The main work we did for this project is located in the  'notebooks' folder. It is what was used to train and test the models. They were ran using Google Collab using their computing resources. The PyTorch code has been adapted to run in python files provided in this repository.

For PyTorch_4Way_C1.py and PyTorch_4Way_C2.py to work you need to download the "data_g_band_v2.txt" file in the following url

https://drive.google.com/drive/folders/1HDShIZNj019MrpXwWl9I748xWLRmgJNl

Paste the "data_g_band_v2.txt" into the "Data" folder and you should be able to run either "PyTorch_4Way_C1"py" or "PyTorch_4Way_C2.py"

run "python PyTorch_4Way_C1.py" to run 4-way classification with PyTorch for C1 network

```sh
! python PyTorch_4Way_C1.py
```

run "python PyTorch_4Way_C2.py" to run 4-way classification with PyTorch for C2 network

```sh
! python PyTorch_4Way_C2.py
```

