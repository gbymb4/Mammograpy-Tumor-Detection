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
    
    rois_data = {'classes': [], 'coords': []}
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


    
def parse_inbreast_xmls(load_limit=None):
    data_dir = f'{c.INBREAST_DIR}/AllXML'
    
    xml_fnames = os.listdir(data_dir)
    xml_fnames = list(filter(lambda x: '.xml' in x, xml_fnames))
    xml_fnames = list(map(lambda x: f'{data_dir}/' + x, xml_fnames))
    xml_fnames = sorted(xml_fnames, key=lambda x: int(Path(x).name.split('_')[0][:-4]))
    
    if load_limit is not None:
        xml_fnames = xml_fnames[:load_limit]
    
    names = np.array([Path(fname).name.split('.')[0] for fname in xml_fnames])
    
    parsed = {name: parse_inbreast_xml(fname) for name, fname in zip(names, xml_fnames)}
    
    return parsed