# -*- coding: utf-8 -*-
"""Implements some utils

TODO:
"""

import random

from torchvision import transforms
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, show_step=10, image_size=30):
        self.transform = transforms.Compose([transforms.Normalize(mean = [-2.118, -2.036, -1.804], # Equivalent to un-normalizing ImageNet (for correct visualization)
                                                                    std = [4.367, 4.464, 4.444]),
                                            transforms.ToPILImage(),
                                            transforms.Scale(image_size)])

        self.show_step = show_step
        self.step = 0

        self.figure, (self.lr_plot, self.hr_plot, self.fake_plot) = plt.subplots(1,3)
        self.figure.show()

        self.lr_image_ph = None
        self.hr_image_ph = None
        self.fake_hr_image_ph = None

    def show(self, inputsG, inputsD_real, inputsD_fake):

        self.step += 1
        if self.step == self.show_step:
            self.step = 0

            i = random.randint(0, inputsG.size(0) -1)

            lr_image = self.transform(inputsG[i])
            hr_image = self.transform(inputsD_real[i])
            fake_hr_image = self.transform(inputsD_fake[i])

            if self.lr_image_ph is None:
                self.lr_image_ph = self.lr_plot.imshow(lr_image)
                self.hr_image_ph = self.hr_plot.imshow(hr_image)
                self.fake_hr_image_ph = self.fake_plot.imshow(fake_hr_image)
            else:
                self.lr_image_ph.set_data(lr_image)
                self.hr_image_ph.set_data(hr_image)
                self.fake_hr_image_ph.set_data(fake_hr_image)

            self.figure.canvas.draw()
