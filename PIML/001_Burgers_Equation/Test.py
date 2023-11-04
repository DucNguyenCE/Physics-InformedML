#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:03:35 2023

@author: nguyenvanduc
"""
import tensorflow as tf

import numpy as np

EBBC = np.array([[1, 1, 0, 0],
                 [0, 0, 1, 1],
                 [1, 1, 1, 0],
                 [1, 0, 1, 1],
                 [1, 1, 1, 1],
                 [1, 0, 1, 0]]);
BCi =np.random.randint(len(EBBC));
U = EBBC[BCi:BCi+1,:];