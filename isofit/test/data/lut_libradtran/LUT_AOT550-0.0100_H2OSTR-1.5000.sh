#!/usr/bin/bash
export cwd=`pwd`
cd /Users/drt/src/libradtran-2.0.2//test
../bin/uvspec < /Users/drt/src/isofit/isofit/test/data/lut_libradtran/LUT_AOT550-0.0100_H2OSTR-1.5000_alb0.inp > /Users/drt/src/isofit/isofit/test/data/lut_libradtran/LUT_AOT550-0.0100_H2OSTR-1.5000_alb0.out
../bin/uvspec < /Users/drt/src/isofit/isofit/test/data/lut_libradtran/LUT_AOT550-0.0100_H2OSTR-1.5000_alb05.inp > /Users/drt/src/isofit/isofit/test/data/lut_libradtran/LUT_AOT550-0.0100_H2OSTR-1.5000_alb05.out
../bin/uvspec < /Users/drt/src/isofit/isofit/test/data/lut_libradtran/LUT_AOT550-0.0100_H2OSTR-1.5000_alb025.inp > /Users/drt/src/isofit/isofit/test/data/lut_libradtran/LUT_AOT550-0.0100_H2OSTR-1.5000_alb025.out
../bin/zenith -s 0 -q -a 34.139247 -o 118.127521 -y 2017.0 8.0 11.0 18.0 42.0 > /Users/drt/src/isofit/isofit/test/data/lut_libradtran/LUT_AOT550-0.0100_H2OSTR-1.5000.zen
cd $cwd
