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
    
    soup = BeautifulSoup(data, 'xml')
    
    rois = list(
        filter(
            lambda x: x.contents == ['ROIs'],
            soup.findAll('key')
        )
    )[0] \
        .findNextSibling() \
        .findAll('dict')
    
    rois_data = []
    for roi in rois:
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
        coords = np.array(coords)
        
        rois_data.append((name, coords))
    
    return rois_data


    
def parse_inbreast_xmls(load_limit=None):
    data_dir = f'{c.INBREAST_DIR}/AllXML'
    
    xml_fnames = os.listdir(data_dir)
    xml_fnames = list(filter(lambda x: '.xml' in x, xml_fnames))
    xml_fnames = list(map(lambda x: f'{data_dir}/' + x, xml_fnames))
    
    if load_limit is not None:
        xml_fnames = xml_fnames[:load_limit]
    
    parsed = [parse_inbreast_xml(fname) for fname in xml_fnames]
    
    data = np.array(parsed)
    names = np.array([Path(fname).name.split('.')[0] for fname in xml_fnames])
    
    return data, names