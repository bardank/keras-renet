# Deep Residual Networks

This repository contains the keras implementation of ResNet models  (ResNet-50, ResNet-101, and ResNet-152).

* ResNet-v1 
It is the original implementation described in the paper "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385). 

* Resnet-v2 
It is the improved implemenation of original ResNet described in the paper "Identity Mappings in Deep Residual Networks" (https://arxiv.org/pdf/1603.05027.pdf).

The improvement is mainly found in the arrangement of layers in the residual block as shown in following figure.

<img src="https://static.packt-cdn.com/products/9781788629416/graphics/B08956_02_10.jpg"
     alt="Residual block compaarison"
     style="margin-right: 10px;" />

### The prominent changes in ResNet v2 are:

* The use of a stack of 1 × 1 - 3 × 3 - 1 × 1 BN-ReLU-Conv2D.
* Batch normalization and ReLU activation come before 2D convolution