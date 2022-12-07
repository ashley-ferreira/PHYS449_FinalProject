# PHYS449_FinalProject
Replicating machine learning results from paper: https://academic.oup.com/mnras/article/506/1/659/6291200

Link to final presenation: https://docs.google.com/presentation/d/1vw_8BbDLDMyIZf-GEiGgTHhTvtZi8xpYM10-5VsCY60/edit?usp=sharing

Development notes from Ashley:

In this Paper we tried to replicate the results from the above paper for their "C2" network and their old "C1" network using PyTorch and Keras. In the paper they had originally used Keras to run their neural networks. We decided to run the neural networks using PyTorch and Keras to compare the modules.

The main work we did for this project is located in the  'Notebooks' folder. It is what was used to train and test the models. They were ran using Google Collab using their computing resources. 


For PyTorch_4Way_C1.py and PyTorch_4Way_C2.py to work you need to download the "data_g_band_v2.txt" file in the following url

https://drive.google.com/drive/folders/1HDShIZNj019MrpXwWl9I748xWLRmgJNl

Paste the "data_g_band_v2.txt" into the "Data" folder and you should be able to run either "PyTorch_4Way_C1"py" or "PyTorch_4Way_C2.py"

run "python PyTorch_4Way_C1.py" to run 4-way classification with PyTorch for C1 network

run "python PyTorch_4Way_C2.py" to run 4-way classification with PyTorch for C2 network

