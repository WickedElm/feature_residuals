#!/usr/bin/env python

import torch as th
import numpy as np
import pandas as pd


class ToTensor(object):
    def __call__(self, sample):
        return th.from_numpy(sample)
