import torch
import os
import struct

# load all frames(100) from a file
def load_file( filename ):
  print( 'loading ' + filename )
  f = open( filename, 'rb' )

  # 100 frames per file
  ret = torch.zeros( (100, 3, 256, 256), dtype=torch.float32 )
  for frame in range(100):
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

def load_files( files ):
  ret = torch.zeros( (len(files), 100, 3, 256, 256), dtype=torch.float32 )
  for i in range(len(files)):
    ret[i] = load_file( files[i] )
  return ret.reshape( -1, 3, 256, 256 )

def load_directory( dirname ):
  files = os.listdir( dirname )
  for i in range(files):
    files[i] = dirname + '/' + files[i]
  return load_files( files )