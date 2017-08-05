# PyTorchSRGAN
A modern PyTorch implementation of SRGAN

It is deeply based on __Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network__ paper published by the Twitter team (https://arxiv.org/abs/1609.04802) but modernizing it with some new promising discoveries such as DenseNets (https://arxiv.org/abs/1608.06993)

I also try to follow best practices described here: https://github.com/soumith/ganhacks

Still a work in progress for now, but hopefully it will serve as a guide for people implementing somewhat complex GANs with PyTorch.

Contributions are welcome!

## Status

Work in progress. Good results are starting to be visible with the CIFAR-100 dataset.

To start a training session:
```
./train --cuda 
```

## Requirements

* PyTorch
* torchvision
