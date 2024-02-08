#!/usr/bin/python3
import torch
import numpy as np
import matplotlib.pyplot as plt

class WaveEquationNN:
  def __init__(self, c):
    self.c = c
    activation = torch.nn.SiLU()
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
      torch.nn.Linear(20,1)
    )

    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005)

  def forward(self, x, t):
    return self.net( torch.hstack( (x, t) ) )

  def loss( self, x, t, u_answer=None ):
    u = self.net( torch.hstack( (x, t) ) )
    ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    utt = torch.autograd.grad(ut, t, grad_outputs=torch.ones_like(ut), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    f = utt - self.c**2 * uxx
    loss = torch.mean( f**2 )
    if u_answer is not None:
      loss = loss + torch.mean( (u - u_answer)**2 )
    return loss

  def train( self, x, t, u_answer=None ):
    self.optimizer.zero_grad()
    loss = self.loss( x, t, u_answer )
    loss.backward()
    self.optimizer.step()
    return loss.item()

wave = WaveEquationNN(1)

T = 5.0
InitialCount = 256
BoundaryCount = 256
MeshCount = 256

InitialT = torch.zeros( (InitialCount,1), dtype=torch.float32, requires_grad=True )

BoundaryX = np.ones( (BoundaryCount//2,1) )
BoundaryX = np.concatenate( (BoundaryX, -BoundaryX) )
BoundaryX = torch.tensor( BoundaryX, dtype=torch.float32, requires_grad=True )
BoundaryU = torch.zeros_like( BoundaryX, dtype=torch.float32, requires_grad=False )

Epochs = 10000

Loss = [0]*Epochs

for i in range(Epochs):
  l = 0
  wave.optimizer.zero_grad()

  InitialX = torch.rand( size=(InitialCount,1), dtype=torch.float32, requires_grad=True )*2 - 1.0
  InitialU = torch.sin(np.pi*InitialX)
  l = l + wave.loss( InitialX, InitialT, InitialU )

  BoundaryT = torch.rand( size=(BoundaryCount,1), dtype=torch.float32, requires_grad=True )*T
  l = l + wave.loss( BoundaryX, BoundaryT, BoundaryU )

  MeshX = torch.rand( size=(MeshCount,1), dtype=torch.float32, requires_grad=True )*2 - 1.0
  MeshT = torch.rand( size=(MeshCount,1), dtype=torch.float32, requires_grad=True )*T
  l = l + wave.loss( MeshX, MeshT )

  print( i, l.item() )
  l.backward()
  wave.optimizer.step()

  Loss[i] = l.item()



PlotX = np.linspace(-1, 1, 100)
PlotT = np.linspace(0, 10, 500)

PlotT, PlotX = np.meshgrid( PlotT, PlotX )
plotshape = PlotT.shape

PlotU = wave.forward(
  torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32),
  torch.tensor(PlotT.reshape(-1,1), dtype=torch.float32)
)

PlotU = PlotU.detach().numpy().reshape( plotshape )

plt.imshow( PlotU, aspect='auto', extent=[0, 5, -1, 1] )
plt.xlabel( 't' )
plt.ylabel( 'x' )
plt.colorbar()
plt.show()

plt.plot( PlotX[:,0], PlotU[:,0], label='t=0' )
plt.plot( PlotX[:,int(500/10.0*0.25)], PlotU[:,int(500/10.0*(2+0.25))], label='t=2.25' )
plt.plot( PlotX[:,int(500/10.0*0.50)], PlotU[:,int(500/10.0*(2+0.50))], label='t=2.50' )
plt.plot( PlotX[:,int(500/10.0*0.75)], PlotU[:,int(500/10.0*(2+0.75))], label='t=2.75' )
plt.plot( PlotX[:,int(500/10.0)], PlotU[:,int(500/10.0*3)], label='t=3.00' )
plt.plot( PlotX[:,int(500/10.0*2)], PlotU[:,int(500/10.0*4)], label='t=4.00' )
plt.legend()
plt.show()

Loss = np.array(Loss)
plt.plot( Loss )
plt.xlabel( 'Epochs' )
plt.ylabel( 'Loss' )
plt.yscale('log')
plt.legend()
plt.show()