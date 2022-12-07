# PHYS 449 Final Project: Galaxy Morphology Classification with Machine Learning

## Contributors
- Yusuf Ahmed (UWaterloo)
- Alexander Caires (UWaterloo)
- Jordan Ducatel (UWaterloo)
- Ashley Ferreira (UWaterloo)
- Guillaume Hewitt (UWaterloo)

## Introduction

Reproducing results from the 2D Convolutional Neural Networks C1 and C2 from "Morphological classification of galaxies with deep learning: comparing 3-way and 4-way CNNs" by Mitchell K. Cavanagh, Kenji Bekki and Brent A. Groves (https://academic.oup.com/mnras/article/506/1/659/6291200) and exploring their application to the Galaxy10 dataset.

The results of Cavanagh *et al*. are replicated in this repository for the new "C2" network introduced in the paper, and their old "C1" network they used previously, using both PyTorch and Keras for each. Cavanagh *et al*. had originally used Keras to run their neural networks. This repository runs both neural networks using PyTorch and Keras, in order to compare the models further.

The main work completed did for this project is located in the `notebooks` folder. Its contents contain the code and structure used to train and test the models. In order to minimize their completion time the code is set up to be initially run using Google Collab, which utilises their computing resources. However, the files containing the Pytorch code can all be run locally, as the code has been adapted to run in python files provided in this repository.

## Running `main.py`

To run `main.py`, use

```sh
! python main.py data/filename.in data/filename.json
```

For PyTorch_4Way_C1.py and PyTorch_4Way_C2.py to work you need to download the "data_g_band_v2.txt" file in the following url

https://drive.google.com/drive/folders/1HDShIZNj019MrpXwWl9I748xWLRmgJNl

Paste the "data_g_band_v2.txt" into the "Data" folder and you should be able to run either "PyTorch_4Way_C1"py" or "PyTorch_4Way_C2.py"

run "python PyTorch_4Way_C1.py" to run 4-way classification with PyTorch for C1 network

run "python PyTorch_4Way_C2.py" to run 4-way classification with PyTorch for C2 network

