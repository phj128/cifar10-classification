# Cifar-10 Classification


## Introduction

This is my personal project in **Introduction of Artificial Intelligence**.

This project is about image classification about ResNet and FC on Cifar-10 dataset.

## Results

|   Network  |   Mean-Acc   |
|------------|--------------|
|    FC-3    |    55.36     |
|    FC-6    |    53.41     |
|    FC-8    |    51.78     |
|    FC-10   |    35.41     |
|   Res-18   |    91.62     |
|   Res-34   |   **92.25**  |
|   Res-50   |    90.53     |
|   Res-101  |    91.76     |

## Installation

-This code was built on Ubuntu 16.04 with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch](http://pytorch.org/) v1.1. 

-NVIDIA GPU is needed for both training and testing. 

-I used RTX 2080 for training and testing.


0. [Recommended] Create a new conda environment.

    ~~~
    conda create -n cifar10 python=3.6
    ~~~

    And activate the environment.
 
    ~~~
    conda activate cifar10
    ~~~

1. Install pytorch.1.1.

    ~~~
    conda install pytorch=1.1 torchvision
    ~~~

    Or from source
   
    ~~~
    conda install pytorch=1.1 torchvision -c pytorch
    ~~~

2. Install dependencies.
 
    ~~~
    pip install numpy matplotlib opencv-python yacs 
    ~~~

    Or from Tsinghua source:
   
    ~~~
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy matplotlib opencv-python yacs 
    ~~~


## Getting started

### Training
    
Train model named fc3 whose network is fc-3 with batchsize=400

    python train.py batch_size 400 model fc3 network fc_3

Train model named res101 whose network is res-101 with batchsize=200

    python train.py batch_size 200 model res101 network res_101

### Testing
   
Test model res101

    python train.py --test batch_size 400 model res101 network res_101

### Visualization

Visualize outputs of model res101

    python train.py --test --vis batch_size 400 model res101 network res_101

