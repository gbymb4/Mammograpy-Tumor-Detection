# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:06:38 2022

@author: Gavin
"""
'''
INBREAST_LABEL_ENCODING_SCHEME = {
    'calcification': 0,
    'calcifications': 0, # typo in original dataset
    'mass': 1,
    'cluster': 2,
    'spiculated region': 3,
    'espiculated region': 3, # typo in original dataset
    'distortion': 4
    # assymetry and pectoral muscles excluded
}

NUM_CLASSES = 5
'''

INBREAST_LABEL_ENCODING_SCHEME = {
    'mass': 0,
    'spiculated region': 1,
    'espiculated region': 1, # typo in original dataset
    'distortion': 2
}

NUM_CLASSES = 3