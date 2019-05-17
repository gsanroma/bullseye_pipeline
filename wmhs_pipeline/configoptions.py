#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:18:10 2017

@author: shahidm
"""

from os.path import realpath, join, abspath, dirname


# defaults
SCRIPT_PATH = dirname(realpath(__file__))

DM_MODEL_DIR = abspath(join(SCRIPT_PATH, 'model'))
