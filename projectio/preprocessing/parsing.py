# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:32:16 2022

@author: Gavin
"""

import os

from pathlib import Path
from bs4 import BeautifulSoup

import pconfig as c
import numpy as np


def parse_inbreast_xml(fname):
    rois_data = {'classes': [], 'coords': []}
    
    if not os.path.isfile(fname):
        return rois_data
    
    with open(fname, 'r') as file:
        data = file.read()
        
        file.close()
    
    soup = BeautifulSoup(data, 'xml')
    
    rois = list(
        filter(
            lambda x: x.contents == ['ROIs'],
            soup.findAll('key')
        )
    )[0] \
        .findNextSibling() \
        .findAll('dict')

    for roi in rois:
        try:
            name = list(
                filter(
                    lambda x: x.contents == ['Name'],
                    roi.findAll('key')
                )
            )[0] \
                .findNextSibling() \
                .contents[0]
            
            coords = list(
                filter(
                    lambda x: x.contents == ['Point_px'],
                    roi.findAll('key')
                )
            )[0] \
                .findNextSibling() \
                .findAll('string')
                
            def make_tuple(coord):
                return eval(coord.text)
                
            coords = list(map(make_tuple, coords))
            
            rois_data['classes'].append(name)
            rois_data['coords'].append(coords)
            
        except IndexError:... # skip missing data
    
    return rois_data


    
def parse_inbreast_xmls(img_names, load_limit=None):
    data_dir = f'{c.INBREAST_DIR}/AllXML'
    
    if load_limit is not None:
        img_names = img_names[:load_limit]
    
    xml_fnames = [f'{data_dir}/{name}.xml' for name in img_names]
    
    names = np.array([Path(fname).name.split('.')[0] for fname in xml_fnames])
    
    parsed = {name: parse_inbreast_xml(fname) for name, fname in zip(names, xml_fnames)}
    
    return parsed
