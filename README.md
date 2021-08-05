# Large-scale-sparse-neural-networks
Proof of concept implementations associated with https://arxiv.org/abs/2102.01732, 
2 February 2021 

![Alt text](graphical_overview.png?raw=true "Title")

## Abstract
Recently, sparse training methods have started to be established as a de facto approach for 
training and inference efficiency in artificial neural networks. Yet, this efficiency is just in theory. 
In practice, everyone uses a binary mask to simulate sparsity since the typical deep learning software 
and hardware are optimized for dense matrix operations.  In this paper, we take an orthogonal approach, 
and we show that we can train truly sparse neural networks to harvest their full potential.  To achieve 
this goal, we introduce three novel contributions, specially designed for sparse neural networks:  **(1)** a 
parallel training algorithm and its corresponding sparse implementation from scratch, **(2)** 
an activation function with non-trainable parameters to favour the gradient flow, and **(3)** a 
hidden neurons importance metric to eliminate redundancies. All in one, we are able to break the record 
and to train the largest neural network ever trained in terms of representational power – reaching the 
size of a bat’s brain. The results show that our approach has state-of-the-art performance, 
outperforming significantly the sparse training baseline and even the dense training counterpart, 
while being faster and opening the path for an environmentally friendly artificial intelligence era.

## Implementation details
The following environment has been selected to implement the parallel algorithm WASAP-SGD, based on 
the implementation provided by [Anderson et al. (2017)](https://arxiv.org/abs/1712.05878) and 
[Liu et al. (2020b)](https://arxiv.org/abs/1901.09181):
* **Language**: Pure Python 3.7 for quick prototyping, where SciPy and Numpy are employed for sparse matrix 
operations while Numba accelerates some critical part of the code such as backpropagation.
* **Framework**: Message Passing Interface (MPI) standard.
* **Library**: mpi4py, the library is constructed on top of the MPI-1/2 specifications and provides an 
object-oriented interface which directly follows MPI-2 C++ bindings. 
All the experiments (with the exception of the Extreme large sparse MLPs subsection) are executed on a typical laptop with the 
following configuration:
* **Hardware** configuration: CPU Intel Core i7-9750H, 2.60 GHz×6, RAM 32 GB, 
Hard disk 1000 GB, NVIDIA GeForceGTX 1650 4GB.
* **Sofware** used: Windows 10, Python 3.7, Numpy 1.19.1, SciPy 1.4.1, and Numba 0.48.0