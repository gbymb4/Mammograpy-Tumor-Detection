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



def parse_ddsm_info(load_limit=None):
    calc_fnames = [
        'calc_case_description_train_set.csv',
        'calc_case_description_test_set.csv'
    ]
    calc_fnames = [f'{c.DDSM_DIR}/csv/{fname}' for fname in calc_fnames]
    
    calc_df = pd.concat([pd.read_csv(fname) for fname in calc_fnames])
    calc_df['lession'] = ['calcification'] * len(calc_df)
    
    mass_fnames = [
        'mass_case_description_train_set.csv',
        'mass_case_description_test_set.csv'
    ]
    mass_fnames = [f'{c.DDSM_DIR}/csv/{fname}' for fname in mass_fnames]
    
    mass_df = pd.concat([pd.read_csv(fname) for fname in mass_fnames])
    mass_df['lession'] = ['mass'] * len(mass_df)
    
    ddsm_df = pd.concat([calc_df, mass_df])
    ddsm_df = ddsm_df.sort_values(by='patient_id')
    
    ddsm_df = ddsm_df.rename(columns={
        'patient_id': 'PatientID',
        'left or right breast': 'Laterality',
        'image view': 'PatientOrientation'
    })
    
    ddsm_df['MammogramPathSuffix'] = [str(Path(path).parent).split('.')[-1] for path in ddsm_df['image file path']]
    ddsm_df['MaskPathSuffix'] = [str(Path(path).parent).split('.')[-1] for path in ddsm_df['ROI mask file path']]
        
    ddsm_df['Laterality'] = ddsm_df['Laterality'].map({'LEFT': 'L', 'RIGHT': 'R'})

    imgs_df = pd.read_csv(f'{c.DDSM_DIR}/csv/dicom_info.csv')
    
    imgs_df = imgs_df[~imgs_df['PatientID'].str.contains("^P", regex=True, na=False)]
    imgs_df['PatientID'] = imgs_df['PatientID'].str.split('_').str[1:3].str.join('_')
    
    imgs_df = imgs_df[imgs_df['SeriesDescription']\
                      .isin(['full mammogram images', 'ROI mask images'])]
    
    grouped = imgs_df.groupby(by=['PatientID', 'PatientOrientation', 'Laterality'])
    
    imgs_df = grouped.filter(
        lambda x: x['SeriesDescription'].nunique() == 2 
    )
    
    imgs_df['PathSuffix'] = [str(Path(path).parent).split('.')[-1] for path in imgs_df['image_path']]
    
    imgs_df = imgs_df.sort_values(by='PatientID')
    
    masks_df = imgs_df.query('SeriesDescription == "ROI mask images"').rename(columns={'PathSuffix': 'MaskPathSuffix'})
    
    masks_df = pd.merge(masks_df, ddsm_df, on='MaskPathSuffix')
    masks_df = masks_df[['PatientID_x', 'PatientOrientation_x', 'Laterality_x', 'SeriesDescription', 'image_path', 'pathology', 'abnormality type']]
    masks_df['image_path'] = ['data/raw/' + fname[5:] for fname in masks_df['image_path']]
    masks_df = masks_df.rename(columns={
        'PatientID_x': 'PatientID',
        'PatientOrientation_x': 'PatientOrientation',
        'Laterality_x': 'Laterality'
    })
    
    masks_df['mammogram_names'] = [f'{x}_{y}_{z}' for x, y, z in masks_df[['PatientID', 'PatientOrientation', 'Laterality']].values]
    
    masks_df.index = list(range(len(masks_df)))
    
    mammograms_df = imgs_df.query('SeriesDescription == "full mammogram images"').rename(columns={'PathSuffix': 'MammogramPathSuffix'})
    mammograms_df = mammograms_df[['PatientID', 'PatientOrientation', 'Laterality', 'SeriesDescription', 'image_path']]
    mammograms_df['image_path'] = ['data/raw/' + fname[5:] for fname in mammograms_df['image_path']]
    
    mammograms_df['mammogram_names'] = [f'{x}_{y}_{z}' for x, y, z in mammograms_df[['PatientID', 'PatientOrientation', 'Laterality']].values]
    
    mammograms_df.index = list(range(len(mammograms_df)))
    
    return mammograms_df, masks_df
    