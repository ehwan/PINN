import matplotlib.pyplot as plt
import numpy as np
import torch

import autoencoder
import stepper

encoder = autoencoder.CavityAutoEncoder()
encoder.load_state_dict( torch.load( 're100ae.pt' ) )
stepper = stepper.CavityLatentStepper()
stepper.load_state_dict( torch.load( 're100step.pt' ) )

# current state ( velx, vely, density )
state = torch.zeros( size=(1,3,256,256), dtype=torch.float32 )
for x in range(256):
  state[0][0][255][x] = 1.0
for y in range(256):
  for x in range(256):
    state[0][2][y][x] = 1.0

def step():
  global state
  latent = encoder.encoder( state )
  next_latent = stepper( latent )
  next_state = encoder.decoder( next_latent )
  state = next_state

def plot():
  Vx = state[0][0].detach().numpy()
  Vy = state[0][1].detach().numpy()
  plt.quiver( Vx, Vy, np.sqrt(Vx**2 + Vy**2) )
  plt.colorbar()
  plt.show()

for i in range(10):
  plot()
  step()