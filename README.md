# PHYS449_FinalProject
Replicating machine learning results from paper: https://academic.oup.com/mnras/article/506/1/659/6291200

Link to final presenation: https://docs.google.com/presentation/d/1vw_8BbDLDMyIZf-GEiGgTHhTvtZi8xpYM10-5VsCY60/edit?usp=sharing

Development notes from Ashley:
- I've implimented the basic skeleton for all 4 CNNs mentioned in the paper
- Padding and stide values seem to be the default ones (no padding, stride of 1)
- The paper implimented these in keras so I did so too but I can easily convert to pytorch and we should be able to easily use our pytorch tensors with a small conversion, we will discuss this
- hyperparams: categorical cross-entropy loss function to be used for all CNNs, adam optimizer and learning rate of  2 × 10^−4 on their network for sure and I don't think they have any learning rate decay or additional regularization beyond dropout. early stopping was used and batch size is described like "Training was conducted over a maximum of 100 epochs with a batch size of 400 for 100 × 100 data, and a reduced batch size of 200 for the 200 × 200 data due to memory limitations"
- Not noted in diagram but I belive they have added batch normalization layers so I will pop those in too
- Adam only used for C2, Adadelta used for others (adaptive learning rate)