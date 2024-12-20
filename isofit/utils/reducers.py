#! /usr/bin/env python3
#
#  Copyright 2019 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Evan Greenberg, evan.greenberg@jpl.nasa.gov

import numpy as np


def band_mean(vals):
    """
    Wrapper for the np.mean reducer. A wrapper is necessary
    because you need to pass it a row-axis call.
    Args:
        vals: (N-locs x bands) array of pixel values
    Returns:
        (b x 1) array of band-means
    """
    return np.mean(vals, axis=0)


def class_priority(vals, thresh=0.25):
    """
    Wrapper for the priority reducer. This might not be the correct
    place for this reducer. The rules are as follows:
    If pct(n) > thresh: -> class = n
    elif pct(n-1) > thresh: -> class = n-1
    else pct(0) > thresh -> class = 0
    This will mean that the largest int class will take precedence.
    The threshold controls how many pixels in the SUB have to be a class
    for it to be important.
    Args:
        vals: (N-locs) array of pixel classes
    Returns:
        (int) Super pixel class
    """
    # Get class counts
    unique, counts = np.unique(vals, return_counts=True)
    pct_counts = counts / np.sum(counts)

    for u, c in zip(unique, pct_counts):
        if c >= thresh:
            surf_class = u

    return int(surf_class)
