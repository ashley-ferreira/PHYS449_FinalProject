#README DATA FOLDER:
#THis folder contains information and files on the SDSS data used.

Files:
  Nair_2010_Catalog.txt: ASCII Format catalogue for classification of 14034 Galaxy morphology by Nair et al., accessed for creating data_g_band.txt file.
  data_prep.py: Python file used to create data_g_band.txt file. This file does not need to be run since the datafile has been created, It is present for completness.
  Pytorch_data_loading.py: Python file contains function used in creating data batches in pytorch for data_g_band.txt.
  sdss_cutout_1min_v2.sh: Bash script used for SDSS data download.
