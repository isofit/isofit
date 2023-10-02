import numpy as np


def pointIndex(points, point):
    """
    Retrieves the row index for a 2d array "point"
    """
    return np.where(np.all(points == point, axis=1))[0][0]


def extractPoints(ds):
    """
    Retrieves the points from an xarray.Dataset object
    Point names must be coordinates along the point dim only
    """
    a, b = ds.point.coords.values()
    return np.vstack([a.values, b.values]).T
