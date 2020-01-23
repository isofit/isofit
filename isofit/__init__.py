#! /usr/bin/env python
#
#  Copyright 2018 California Institute of Technology
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
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#         Adam Erickson, adam.m.erickson@nasa.gov
#


### Variables ###

name = 'isofit'

__version__ = '1.3.1'

jit_enabled = False

warnings_enabled = False


### Classes ###

class conditional_decorator(object):
    """Decorator class to conditionally apply a decorator to a function definition.

    Attributes
    ----------
    decorator : object
        a decorator to conditionally apply to the funcion
    condition : bool
        a boolean indicating whether the condition is met
    """

    def __init__(self, dec, condition, **kwargs):
        """
        Parameters
        ----------
        dec : object
            a decorator to conditionally apply to the funcion
        condition : bool
            a boolean indicating whether the condition is met
        """

        self.decorator = dec
        self.condition = condition
        self.kwargs = kwargs

    def __call__(self, func):
        """
        Parameters
        ----------
        func : object
            a function to return with or without a decorator

        Returns
        -------
        object
            original function without decorator if condition is unmet
        """

        if not self.condition:
            return func
        return self.decorator(func, **self.kwargs)
