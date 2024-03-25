#! /usr/bin/python3

import torch
import numpy as np
import os
import struct

class CavityAutoEncoder(torch.nn.Module):
  def __init__(self):
    super(CavityAutoEncoder, self).__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d(3, 32, 5, padding=1 ), # 256x256 -> 256x256
      torch.nn.ReLU(),
      torch.nn.Conv2d(32, 32, 5, padding=2 ),
      torch.nn.ReLU(),
      torch.nn.Conv2d(32, 64, 5, padding=2, stride=2 ), # 256x256 -> 128x128
      torch.nn.ReLU(),
      torch.nn.Conv2d(64, 64, 5, padding=2 ),
      torch.nn.ReLU(),
      torch.nn.Conv2d(64, 128, 5, padding=2, stride=4 ), # 128x128 -> 32x32
      torch.nn.ReLU(),
      torch.nn.Conv2d(128, 128, 3, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(128, 256, 3, padding=1, stride=2), # 32x32 -> 16x16
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 256, 3, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 256, 3, padding=1, stride=2), # 16x16 -> 8x8
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 256, 3, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 256, 3, padding=1, stride=2), # 8x8 -> 4x4
      torch.nn.Flatten(),
      torch.nn.Linear( 4096, 1024 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 1024, 128 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 128, 32 )
    )
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear( 32, 128),
      torch.nn.SiLU(),
      torch.nn.Linear( 128, 1024),
      torch.nn.SiLU(),
      torch.nn.Linear( 1024, 4096),
      torch.nn.SiLU(),
      torch.nn.Unflatten(1, (256, 4, 4)),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 256, 256, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 256, 256, 3, padding=1, stride=2, output_padding=1 ), # 4x4 -> 8x8
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 256, 256, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 256, 256, 3, padding=1, stride=2, output_padding=1 ), # 8x8 -> 16x16
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 256, 256, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 256, 128, 3, padding=1, stride=2, output_padding=1 ), # 16x16 -> 32x32
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 128, 128, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 128, 64, 5, padding=2, stride=4, output_padding=3 ), # 32x32 -> 128x128
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 64, 64, 5, padding=2 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 64, 32, 5, padding=2, stride=2, output_padding=1 ), # 128x128 -> 256x256
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 32, 32, 5, padding=2 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 32, 3, 5, padding=2 )
    )


def load_images( filename ):
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

def load_all_images():
  directory = 'trains'
  files = os.listdir(directory)
  inputs = torch.zeros( (len(files), 101, 3, 256, 256), dtype=torch.float32 )
  for i in range(len(files)):
    inputs[i] = load_images( directory + '/' + files[i] )
  return inputs.reshape( -1, 3, 256, 256 )

autoencoder = CavityAutoEncoder()
inputs = load_images( 'trains/re100.dat' )
print( inputs.shape )

N = inputs.shape[0]

Epochs = 50
BatchSize = 20
optimizer = torch.optim.Adam( autoencoder.parameters(), lr=0.001 )
for epoch in range(Epochs):
  print( 'Epoch: {}'.format(epoch) )
  shuffled = inputs[ torch.randperm(N) ]
  for batch in range(0, N, BatchSize):
    x = shuffled[batch:batch+BatchSize]
    print( 'train batch: ', x.shape )
    latent = autoencoder.encoder( x )
    decoded = autoencoder.decoder( latent )
    loss = torch.nn.functional.mse_loss( decoded, x )
    print( 'Loss: {}'.format(loss.item()) )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# test_input = torch.rand(1, 3, 256, 256)
# latent = nn.encoder( test_input )
# print( latent.shape )
# decoded = nn.decoder( latent )
# print( decoded.shape )