# PyTorchSRGAN
A modern PyTorch implementation of SRGAN

It is deeply based on __Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network__ paper published by the Twitter team (https://arxiv.org/abs/1609.04802) but I replaced activations by Swish (https://arxiv.org/abs/1710.05941)

Experiments are being done with the CIFAR-100 dataset, due to computational limitations, but I hope to use larger images later on.

Contributions are welcome!

## Training

```
usage: train [-h] [--dataset DATASET] [--dataroot DATAROOT]
             [--workers WORKERS] [--batchSize BATCHSIZE]
             [--imageSize IMAGESIZE] [--upSampling UPSAMPLING]
             [--nEpochs NEPOCHS] [--generatorLR GENERATORLR]
             [--discriminatorLR DISCRIMINATORLR] [--cuda] [--nGPU NGPU]
             [--generatorWeights GENERATORWEIGHTS]
             [--discriminatorWeights DISCRIMINATORWEIGHTS] [--out OUT]
```

Example: ```./train --cuda```

This will start a training session in the GPU. First it will pre-train the generator using MSE error for 300 epochs, then it will train the full GAN (generator + discriminator) for 100 epochs, using content (mse + vgg) and adversarial loss. Although weights are already provided in the repository, this script will also generate them in the checkpoints file.

## Testing

```
usage: test [-h] [--dataset DATASET] [--dataroot DATAROOT] [--workers WORKERS]
            [--batchSize BATCHSIZE] [--imageSize IMAGESIZE]
            [--upSampling UPSAMPLING] [--cuda] [--nGPU NGPU]
            [--generatorWeights GENERATORWEIGHTS]
            [--discriminatorWeights DISCRIMINATORWEIGHTS]

```

Example: ```./test --cuda```

This will start a testing session in the GPU. It will display mean error values and save the first image of each batch in the output directory, all three versions: low resolution, high resolution (original) and high resolution (generated).

## Requirements

* PyTorch
* torchvision
* tensorboard_logger (https://github.com/TeamHG-Memex/tensorboard_logger)
