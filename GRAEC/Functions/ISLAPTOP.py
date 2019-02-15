# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:55:21 2018

@author: Yorick Spenrath
"""
"""
Functionality to handle working on different devices
"""

import os


def username():
    dirs = os.listdir('C:/Users')
    options = ['Yorick Spenrath', '179040', 's118344']
    result = None
    for op in options:
        if op in dirs:
            if result is None:
                result = op
            else:
                raise Exception('Multiple options for username')
    if result is None:
        raise Exception('No options for username')
    return result
