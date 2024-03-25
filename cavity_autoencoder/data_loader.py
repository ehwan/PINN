import torch
import os
import struct

# load first N (~100) frames from a file
def load_file( filename, N=100 ):
  print( 'loading ' + filename )
  f = open( filename, 'rb' )

  # 100 frames per file
  ret = torch.zeros( (N, 3, 256, 256), dtype=torch.float32 )
  for frame in range(N):
    print( 'frame ' + str(frame) )
    # vels
    buffer = f.read( 4*2*256*256 )
    vel = struct.unpack( 'ff'*256*256, buffer )
    for y in range(256):
      for x in range(256):
        idx = y*256 + x
        ret[frame, 0, y, x] = vel[idx*2]
        ret[frame, 1, y, x] = vel[idx*2+1]
    # density
    buffer = f.read( 4*256*256 )
    dens = struct.unpack( 'f'*256*256, buffer )
    for y in range(256):
      for x in range(256):
        idx = y*256 + x
        ret[frame, 2, y, x] = dens[idx]

  return ret

def load_files( files, N=100 ):
  ret = torch.zeros( (len(files), N, 3, 256, 256), dtype=torch.float32 )
  for i in range(len(files)):
    ret[i] = load_file( files[i], N )
  return ret.reshape( -1, 3, 256, 256 )

def load_directory( dirname, N=100 ):
  files = os.listdir( dirname )
  for i in range(files):
    files[i] = dirname + '/' + files[i]
  return load_files( files, N )