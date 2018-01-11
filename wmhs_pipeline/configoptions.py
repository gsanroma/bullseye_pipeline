#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:18:10 2017

@author: shahidm
"""

from os.path import realpath, join, abspath, dirname


# defaults
SCRIPT_PATH = dirname(realpath(__file__))

WMHS_MASKS_DIR = abspath(join(SCRIPT_PATH, 'WMHmaskbin'))

BIANCA_CLASSIFIER_DATA=abspath(join(SCRIPT_PATH, 'WMHmaskbin', 'classif17manual_labels.dat'))
BIANCA_CLASSIFIER_LABELS=abspath(join(SCRIPT_PATH, 'WMHmaskbin', 'classif17manual_labels.dat_labels'))
