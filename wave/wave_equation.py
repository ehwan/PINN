#!/usr/bin/python3
import torch
import numpy as np
import matplotlib.pyplot as plt

class WaveEquationNN:
  def __init__(self, c):
    self.c = c
    activation = torch.nn.SiLU() # Swish
    self.net = torch.nn.Sequential(
      torch.nn.Linear(2, 20),
      activation,
      torch.nn.Linear(20, 20),
      activation,
      torch.nn.Linear(20, 20),
      activation,
      torch.nn.Linear(20, 20),
      activation,
      torch.nn.Linear(20, 20),
      activation,
      torch.nn.Linear(20, 20),
      activation,
      torch.nn.Linear(20, 20),
      activation,
      torch.nn.Linear(20, 20),
      activation,
      torch.nn.Linear(20, 20),
      activation,
      torch.nn.Linear(20,1)
    )

    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005)

  def forward(self, x, t):
    return self.net( torch.hstack( (x, t) ) )

  def function( self, x, t ):
    u = self.net( torch.hstack( (x, t) ) )
    ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    utt = torch.autograd.grad(ut, t, grad_outputs=torch.ones_like(ut), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    f = utt - self.c**2 * uxx
    return u, ux, ut, uxx, utt, f

  def loss( self, x, t, u_answer=None ):
    u, ux, ut, uxx, utt, f = self.function(x,t)

    l = 0

    # L2 norm of Physics Informed
    l = l + torch.mean( f**2 )
    # L Infinity norm
    # l = l + f.abs().max()
    if u_answer is not None:
      l = l + torch.mean( (u - u_answer)**2 )
    return l

wave = WaveEquationNN(1)
# wave.net.load_state_dict( torch.load('wave_equation.pt') )

InitialN = 128
InitialT = torch.zeros( size=(InitialN,1), dtype=torch.float32, requires_grad=True )
BoundaryN = 128
BoundaryX = torch.ones( size=(BoundaryN//2,1), dtype=torch.float32, requires_grad=True )
BoundaryX = torch.vstack( (BoundaryX, -BoundaryX) )
BoundaryU = torch.zeros_like(BoundaryX)
MeshN = 128

T = 8.0

Epochs = 50000

Loss = [0]*Epochs

for i in range(Epochs):
  l = 0
  wave.optimizer.zero_grad()

  InitialX = torch.rand( size=(InitialN,1), dtype=torch.float32, requires_grad=True )*2 - 1.0
  InitialU = torch.sin( np.pi * InitialX )
  l = l + wave.loss( InitialX, InitialT, InitialU )

  MeshX = torch.rand( size=(MeshN,1), dtype=torch.float32, requires_grad=True )*2 - 1.0
  MeshT = torch.rand( size=(MeshN,1), dtype=torch.float32, requires_grad=True )*T
  l = l + wave.loss( MeshX, MeshT )

  BoundaryT = torch.rand( size=(BoundaryN,1), dtype=torch.float32, requires_grad=True )*T
  l = l + wave.loss( BoundaryX, BoundaryT, BoundaryU )

  print( i, l.item() )
  l.backward()
  wave.optimizer.step()

  Loss[i] = l.item()

torch.save(wave.net.state_dict(), 'wave_equation.pt')



PlotX = np.linspace(-1, 1, 100)
PlotT = np.linspace(0, 10, 500)

PlotT, PlotX = np.meshgrid( PlotT, PlotX )
plotshape = PlotT.shape

PlotU, _, _, _, _, PlotF = wave.function(
  torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32, requires_grad=True),
  torch.tensor(PlotT.reshape(-1,1), dtype=torch.float32, requires_grad=True)
)

PlotU = PlotU.detach().numpy().reshape( plotshape )
PlotF = PlotF.detach().numpy().reshape( plotshape )

plt.imshow( PlotU, aspect='auto', extent=[0, 10, -1, 1], vmin=-1.5, vmax=1.5 )
plt.xlabel( 't' )
plt.ylabel( 'x' )
plt.colorbar()
plt.show()

plt.imshow( PlotF, aspect='auto', extent=[0, 10, -1, 1] )
plt.xlabel( 't' )
plt.ylabel( 'x' )
plt.colorbar()
plt.show()

Loss = np.array(Loss)
plt.plot( Loss )
plt.xlabel( 'Epochs' )
plt.ylabel( 'Loss' )
plt.yscale('log')
plt.legend()
plt.show()
