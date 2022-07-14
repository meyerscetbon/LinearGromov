# Linear-Time Gromov Wasserstein Distances using Low Rank Couplings and Costs
Code of the [paper](https://arxiv.org/pdf/2106.01128.pdf) by Meyer Scetbon, Gabriel Peyré and Marco Cuturi.



## A New Way to Regularize the GW Problem
In this work, we propose to regularize the Gromov-Wasserstein (GW) problem by constraining the admissible couplings to have a low-nonnegative rank: we call it the Low-Rank Gromov-Wasserstein (LR-GW) Problem. In the following figure, we compare the couplings obtained by the Prior Art method based on an entropic regularization and ours.
![figure](figures/coupling_plot.jpg)


Our regularization can take also advantage of the geometry of the problem, in particular when the cost matrices involved in the GW problem admits a low-rank factorization. In this case, our method is able to compute the the LR-GW cost in linear time with respect to the number of samples. We present the time-accuracy tradeoff between different methods when the samples are drawn from two anisotropic Gaussian blobs of 5 or 20 clusters in 10D and 15D,
endowed with the squared Euclidean distance  with n = m = 10000 samples. 
![figure](figures/acc_plot.jpg)

In this [file](https://github.com/meyerscetbon/LinearGromov/blob/main/toy_examples.py) we provide some toy examples where we compare the Entropic GW scheme with our proposed method. 

This repository contains a Python implementation of the algorithms presented in the [paper](https://arxiv.org/pdf/2106.01128.pdf).
