#!/usr/bin/env python

import argparse
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import Generator, Discriminator, FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=15, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='checkpoints/generator_final.pth', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='checkpoints/discriminator_final.pth', help="path to discriminator weights (to continue training)")

opt = parser.parse_args()
print(opt)

try:
    os.makedirs('output/high_res_fake')
    os.makedirs('output/high_res_real')
    os.makedirs('output/low_res')
except OSError:
    pass


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize*opt.upSampling),
                                transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

# Equivalent to un-normalizing ImageNet (for correct visualization)
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

if opt.dataset == 'folder':
    # folder dataset
    dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
elif opt.dataset == 'cifar10':
    dataset = datasets.CIFAR10(root=opt.dataroot, download=True, train=False, transform=transform)
elif opt.dataset == 'cifar100':
    dataset = datasets.CIFAR100(root=opt.dataroot, download=True, train=False, transform=transform)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

generator = Generator(16, opt.upSampling)
if opt.generatorWeights != '':
    generator.load_state_dict(torch.load(opt.generatorWeights))
print generator

discriminator = Discriminator()
if opt.discriminatorWeights != '':
    discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
print discriminator

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print feature_extractor
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

target_real = Variable(torch.ones(opt.batchSize,1))
target_fake = Variable(torch.zeros(opt.batchSize,1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    target_real = target_real.cuda()
    target_fake = target_fake.cuda()

low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

print 'Test started...'
mean_generator_content_loss = 0.0
mean_generator_adversarial_loss = 0.0
mean_generator_total_loss = 0.0
mean_discriminator_loss = 0.0

# Set evaluation mode (not training)
generator.eval()
discriminator.eval()

for i, data in enumerate(dataloader):
    # Generate data
    high_res_real, _ = data

    # Downsample images to low resolution
    for j in range(opt.batchSize):
        low_res[j] = scale(high_res_real[j])
        high_res_real[j] = normalize(high_res_real[j])

    # Generate real and fake inputs
    if opt.cuda:
        high_res_real = Variable(high_res_real.cuda())
        high_res_fake = generator(Variable(low_res).cuda())
    else:
        high_res_real = Variable(high_res_real)
        high_res_fake = generator(Variable(low_res))
    
    ######### Test discriminator #########

    discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                            adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
    mean_discriminator_loss += discriminator_loss.data[0]

    ######### Test generator #########

    real_features = Variable(feature_extractor(high_res_real).data)
    fake_features = feature_extractor(high_res_fake)

    generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
    mean_generator_content_loss += generator_content_loss.data[0]
    generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), target_real)
    mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

    generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
    mean_generator_total_loss += generator_total_loss.data[0]

    ######### Status and display #########
    sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (i, len(dataloader),
    discriminator_loss.data[0], generator_content_loss.data[0], generator_adversarial_loss.data[0], generator_total_loss.data[0]))

    for j in range(opt.batchSize):
        save_image(unnormalize(high_res_real.data[j]), 'output/high_res_real/' + str(i*opt.batchSize + j) + '.png')
        save_image(unnormalize(high_res_fake.data[j]), 'output/high_res_fake/' + str(i*opt.batchSize + j) + '.png')
        save_image(unnormalize(low_res[j]), 'output/low_res/' + str(i*opt.batchSize + j) + '.png')

sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (i, len(dataloader),
mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))