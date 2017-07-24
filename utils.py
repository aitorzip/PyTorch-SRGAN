# -*- coding: utf-8 -*-
"""Implements some utils

TODO:
"""

import random

from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, show_step=10, image_size=(120,120)):
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Scale(image_size)])
        self.show_step = show_step
        self.step = 0

        self.figure, (self.lr_plot, self.hr_plot, self.fake_plot) = plt.subplots(1,3)
        self.figure.show()

    def show(self, inputsG, inputsD_real, inputsD_fake):

        self.step += 1
        if self.step == self.show_step:
            self.step = 0

            i = random.randint(0, inputsG.size(0) -1)

            lr_image = self.transform(inputsG[i])
            hr_image = self.transform(inputsD_real[i])
            fake_hr_image = self.transform(inputsD_fake[i])

            self.lr_plot.imshow(lr_image)
            self.hr_plot.imshow(hr_image)
            self.fake_plot.imshow(fake_hr_image)
            self.figure.canvas.draw()
