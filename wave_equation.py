#!/usr/bin/python3
import torch
import numpy as np
import matplotlib.pyplot as plt

class WaveEquationNN:
  def __init__(self, c):
    self.c = c
    self.net = torch.nn.Sequential(
      torch.nn.Linear(2, 20),
      torch.nn.Tanh(),
      torch.nn.Linear(20, 20),
      torch.nn.Tanh(),
      torch.nn.Linear(20, 20),
      torch.nn.Tanh(),
      torch.nn.Linear(20, 20),
      torch.nn.Tanh(),
      torch.nn.Linear(20,1)
    )

    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

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

X0 = np.linspace(-1, 1, 100).reshape( -1, 1 )
U0 = np.sin(np.pi*X0)

X0 = torch.tensor( X0, dtype=torch.float32, requires_grad=True )
U0 = torch.tensor( U0, dtype=torch.float32, requires_grad=True )
T0 = torch.zeros_like(X0, requires_grad=True)

BoundT = np.random.rand(100).reshape( -1, 1 )
BoundT = torch.tensor( BoundT, dtype=torch.float32, requires_grad=True )
BoundU = torch.zeros_like(BoundT, requires_grad=True)
BoundX1 = torch.ones_like(BoundT, requires_grad=True)
BoundX2 = -torch.ones_like(BoundT, requires_grad=True)

MeshX = np.linspace(-1, 1, 100)
MeshT = np.linspace(0, 4, 500)
MeshX, MeshT = np.meshgrid( MeshX, MeshT )
MeshU = np.sin(np.pi*MeshX) * np.cos(np.pi*MeshT)

MeshX = torch.tensor( MeshX.reshape(-1,1), dtype=torch.float32, requires_grad=True )
MeshT = torch.tensor( MeshT.reshape(-1,1), dtype=torch.float32, requires_grad=True )
MeshU = torch.tensor( MeshU.reshape(-1,1), dtype=torch.float32, requires_grad=True )

for i in range(5000):
  # print( wave.train( X0, T0, U0 ) )
  # print( wave.train( BoundX1, BoundT, BoundU ) )
  # print( wave.train( BoundX2, BoundT, BoundU ) )
  print( wave.train( MeshX, MeshT, MeshU ) )



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
plt.plot( PlotX[:,int(500/5.0*0.25)], PlotU[:,int(500/5.0*0.25)], label='t=0.25' )
plt.plot( PlotX[:,int(500/5.0*0.50)], PlotU[:,int(500/5.0*0.50)], label='t=0.50' )
plt.plot( PlotX[:,int(500/5.0*0.75)], PlotU[:,int(500/5.0*0.75)], label='t=0.75' )
plt.plot( PlotX[:,int(500/5.0)], PlotU[:,int(500/5.0)], label='t=1.00' )
plt.legend()
plt.show()