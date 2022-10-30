# PHYS449_FinalProject
Replicating machine learning results from paper: https://academic.oup.com/mnras/article/506/1/659/6291200

Google drive: https://drive.google.com/drive/folders/1199S9kkTg9Qf7ZIUdW3EBEGS5hGc8MgG?usp=sharing


Development notes from Ashley:
- I've implimented the basic skeleton for all 4 CNNs mentioned in the paper
- I will need to review aspects to make sure they are correct and see if their is indication of padding and stide values that are different than the default ones
- The paper implimented these in keras so I did so too but I can easily convert to pytorch and we should be able to easily use our pytorch tensors with a small conversion, we will discuss this
- hyperparams: categorical cross-entropy loss function to be used for all CNNs, adam optimizer and learning rate of  2 × 10^−4 for 20 epochs on their network for sure and I don't think they have any learning rate decay or additional regularization beyond dropout. Couldn't find batch size.
- Not noted in diagram but I belive they have added batch normalization layers so I will pop those in too