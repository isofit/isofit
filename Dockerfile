FROM osgeo/gdal:ubuntu-full-3.4.0

RUN apt-get update
RUN apt-get install python3-pip git -y

RUN apt-get install -y libnetcdf-dev libnetcdff-dev libgsl-dev
RUN apt-get install gfortran -y

RUN apt-get install python2 -y
