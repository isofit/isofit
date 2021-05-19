#!/usr/bin/env python
# David R Thompson
# Clip the dark current segments at the start and end of an image

import os
import sys
import spectral
import argparse
from scipy import *
from scipy import linalg


chunksize = 500  # process this many lines at once

# Return the header associated with an image file
def find_header(imgfile):
  if os.path.exists(imgfile+'.hdr'):
    return imgfile+'.hdr'
  ind = imgfile.rfind('.raw')
  if ind >= 0:
    return imgfile[0:ind]+'.hdr'
  ind = imgfile.rfind('.img')
  if ind >= 0:
    return imgfile[0:ind]+'.hdr'
  raise IOError('No header found for file {0}'.format(imgfile));

def apply_glt(in_file, glt_file, out_file):
  in_hdr = find_header(in_file)
  #print (in_hdr)
  if not os.path.exists(in_hdr):
    raise IOError('cannot find a header');
  img = spectral.io.envi.open(in_hdr, in_file)

  '''
  - The GLT contains integer pixel coordinates (row,col) that map directly into the IGM, with any negative (row,col) index values indicating that the pixel at that particular location was filled via nearest-neighbor interpolation, and should be populated with the value at index (abs(row),abs(col)). 

  - The rotation parameter is associated with the(utm_x,utm_y) map coordinates for each pixel coordinate in the GLT. The  map coords are computed for each (row,col) pixel coord in the GLT wrt the map info metadata -- specifically the upper left UTM map coordinate (ulx,uly) and pixel size ps, specified in the map info string in the .hdr, and then those coordinates are rotated to get the correct ground location.  
  '''

  glt_hdr = find_header(glt_file)
  #print (glt_hdr)
  #input()
  if not os.path.exists(glt_hdr):
    raise IOError('cannot find a header');
  glt = spectral.io.envi.open(glt_hdr, glt_file)
  rows = abs(glt.read_band(1)) 
  cols = abs(glt.read_band(0))
  bad = (rows==0) | (cols == 0)

  rows = rows - 1 # convert to zero-based indexing
  cols = cols - 1 # also zero indexing
    
  print ("len(glt.read_band(0)), len(glt.read_band(1))")
  print (len(glt.read_band(0)), len(glt.read_band(1)))
  print ("glt.read_band(0).shape {} glt.read_band(1).shape {}".format(glt.read_band(0).shape, glt.read_band(1).shape))
  print ("rows {}\ncols {}\n".format(rows, cols))
  
  flg = rows >= 0
  print(rows[flg].shape)
  print("rows {}".format(rows[flg]))
  print("min(rows) {}".format(min(rows[flg])))
  print("largest(rows) {}".format(max(rows[flg])))
  print("min(cols) {}".format(min(cols[flg])))
  print("largest(cols) {}".format(max(cols[flg])))
  
  out_hdr = out_file+'.hdr'

  # Get metadata
  metadata = img.metadata.copy()

  
  interleave = img.metadata['interleave']

  for field in ['lines','samples']:
    metadata[field] = glt.metadata[field]

  # format map info string properly because ENVI makes poor default decisions
  metadata['map info'] = '{%s}'%(', '.join(glt.metadata['map info']))

  metadata['data ignore value'] = -9999
  out = spectral.io.envi.create_image(out_hdr, metadata, ext='', force=True)
  outmm = out.open_memmap(interleave='source', writable=True)
  
  nl = int(metadata['lines'])
  nstep = 10
  step = int(nl/nstep)
  iper = 0
  for i in range(nl):
    if i % step == 0 or i==nl:
      print('Processing line %i of %i   \t%3d%%'%(i,nl,100*float(iper)/nstep))
      iper = iper+1
      
    outmm[i,:,:] = 0

    if i==0 or i%100==0:
    #if i==0: # yumi
      del img
      del out
      img = spectral.io.envi.open(in_hdr, in_file)
      inmm = img.open_memmap(interleave='source', writable=False)
      out = spectral.io.envi.open(out_hdr, out_file)
      outmm = out.open_memmap(interleave='source', writable=True)
      #print (i)
      #print ("inmm.shape {}".format(inmm.shape))
      #print ("outmm.shape {}".format(outmm.shape))


    # cols for igm  are 0-605 -> should be the same off_pb with rdn file

    # spatial subset
    if interleave == 'bil':
      for j,(r,c,b) in enumerate(zip(rows[i,:], cols[i,:], bad[i,:])):
        # if b == False: print (j,(r,c,b))
        #if i > 4842: print j,r,c,b # debug

        if not b:
          outmm[i,:,j] = array(inmm[r,:,c],dtype=float32)
        else:
          outmm[i,:,j] = -9999

    elif interleave == 'bip': 
      for j,(r,c,b) in enumerate(zip(rows[i,:], cols[i,:], bad[i,:])):

        if not b:
          outmm[i,j,:] = array(inmm[r,c,:],dtype=float32)
        else:
          outmm[i,j,:] = -9999

    else:
      raise ValueError('cannot use %s interleave'%img.metadata['interleave'])

  del out 
  del glt
  del img
 

def main():
  # parse the command line (perform the correction on all command line arguments)
  parser = argparse.ArgumentParser(description="Clip dark lines from file")
  parser.add_argument("in_file", metavar = "IMAGE", type=str, 
      help="Radiance image")
  parser.add_argument("glt_file", metavar = "GLT", type=str, 
      help="GLT image")
  parser.add_argument("out_file", metavar = 'OUT', type=str, 
      help="output")
 
  args = parser.parse_args(sys.argv[1:])

  in_file = args.in_file
  glt_file = args.glt_file
  out_file = args.out_file

  apply_glt(in_file,glt_file,out_file)

if __name__ == "__main__":
  main()

