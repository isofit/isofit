#! /usr/bin/env python3
#
# Author: Niklas Bohn
#

from mlky import register


@register()
def check_sensor_name(value):
    """Check if valid sensor name is given in config file."""
    if value not in ["ang", "avcl", "prism", "neon", "emit", "NA-*", "hyp"]:
        return (
            "No valid sensor name given in config file. Please provide valid sensor name in config file (choose "
            "from 'ang', 'avcl', 'prism', 'neon', 'emit', 'NA-*', 'hyp')"
        )
    return True
