# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 18:37:37 2023

@author: Gavin
"""

def copy_attributes(from_instance, to_instance):
    for attr_name in dir(from_instance):
        if not attr_name.startswith("__"):
            attr_value = getattr(from_instance, attr_name)
            setattr(to_instance, attr_name, attr_value)