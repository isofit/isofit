##################################################################
#
#  Script file to download and extract example datasets
#
#
##################################################################

# if re-running, cleanup older data
DATA_DIR=hypertrace-data
if [ -d "$DATA_DIR" ]; then rm -Rf $DATA_DIR; fi

# download data
data_source_url=https://github.com/ashiklom/isofit/releases/download/hypertrace-data/hypertrace-data.tar.gz
wget ${data_source_url}

# unpack
tar -zxvf hypertrace-data.tar.gz

# cleanup, remove tar.gz file
rm hypertrace-data.tar.gz
