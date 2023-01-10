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
import pandas as pd

def parse_inbreast_xml(fname, cull_singles=True):
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
                
            if cull_singles and len(coords) == 1:
                continue
                
            def make_tuple(coord):
                return eval(coord.text)
                
            coords = list(map(make_tuple, coords))
            
            rois_data['classes'].append(name)
            rois_data['coords'].append(coords)
            
        except IndexError:... # skip missing data
    
    return rois_data


    
def parse_inbreast_xmls(img_names, load_limit=None, **kwargs):
    data_dir = f'{c.INBREAST_DIR}/AllXML'
    
    if load_limit is not None:
        img_names = img_names[:load_limit]
    
    xml_fnames = [f'{data_dir}/{name}.xml' for name in img_names]
    
    names = np.array([Path(fname).name.split('.')[0] for fname in xml_fnames])
    
    parsed = {name: parse_inbreast_xml(fname, **kwargs) for name, fname in zip(names, xml_fnames)}
    
    return parsed



def parse_mias_info(load_limit=None, cull_no_rad=True):
    fname = f'{c.MIAS_DIR}/Info.txt'
    
    with open(fname, 'r') as file:
        content = file.readlines()

        file.close()
        
    col_names = content[0].strip().split()
    
    data = content[1:]
    
    if load_limit is not None:
        data = data[:load_limit]
    
    rows = list(map(str.split, data))
    
    parsed = pd.DataFrame(rows, columns=col_names)
    
    if cull_no_rad:
        parsed = parsed.dropna()
    
    return parsed