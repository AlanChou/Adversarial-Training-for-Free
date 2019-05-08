# Adversarial Training for Free

This is an unofficial PyTorch implementation of the paper "Adversarial Training for Free!".<br> 
https://128.84.21.199/pdf/1904.12843.pdf<br>
It's a really helpful technique which can significantly accelerate adversarial training. <br>
It only contains the code for adversarial training on CIFAR-10. However, one can easily modify it for other datasets. 

I've noticed that in the cleverhans repo, they have an attribute called BACKPROP_THROUGH_ATTACK which is exactly the same idea behind this paper. 

## Overview

This repository contains two files. One for adversarial training and one for testing. Note that the model used here is not WideResNet 32-10 which is used in the paper 'Adversarial Training for Free'. I use WideResNet 28-10 which is used in the original PGD paper. Be aware that some of the hyperparameters are slightly different from the paper (weight decay and learning rate scheduling)



## Accuracy (under PGD attack: epsilon = 8, step size = 2, iteration = 100)
| Model                      | Acc         |
| ---------------------------| ----------- |
| [ WideResNet 28-10 ]       | 46.93%      |

I did not test every epoch's checkpoint. I simply chose from one of the last epochs to test. Results might be slightly different. I've also released the checkpoint in the below Google Drive link) <br>
I have trouble training with ResNet56 and ResNet20 to the baseline they supposed to have which may suggest that this method does not apply to any given models. Please don't hesitate to share your results if you know how to fix this.
`checkpoint` [Google Drive](https://drive.google.com/file/d/1iZ52Ctcwty8bLMvLJJMlWcHL-__lJcbo/view?usp=sharing) 

## Dependencies
The repository is based on Python 3.5, with the main dependencies being PyTorch==1.0.0 Additional dependencies for running experiments are: numpy, tqdm, argparse, os, random, advertorch.

Advertorch can be installed from https://github.com/BorealisAI/advertorch<br>
Run the code with the command:<br>
```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py 
```


