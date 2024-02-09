#!/usr/bin/python3
import torch
import numpy as np
import matplotlib.pyplot as plt

class BurgerEquationNN(torch.nn.Module):
  def __init__(self, nu):
    super().__init__()
    self.nu = nu
    # activation = torch.nn.Tanh()
    activation = torch.nn.SiLU()
    # activation = torch.nn.LeakyReLU()
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
    # self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=0.003)

  def forward(self, x, t):
    return self.net( torch.hstack( (x, t) ) )

  def loss( self, x, t, u_answer=None ):
    u = self.net( torch.hstack( (x, t) ) )
    ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    f = ut + u * ux - self.nu * uxx
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

burger = BurgerEquationNN(0.01/np.pi)
try:
  burger.net.load_state_dict( torch.load('burger.pt') )
except FileNotFoundError:
  pass

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
  burger.optimizer.zero_grad()

  InitialX = torch.rand( size=(InitialCount,1), dtype=torch.float32, requires_grad=True )*2 - 1.0
  InitialU = -torch.sin(np.pi*InitialX)
  l = l + burger.loss( InitialX, InitialT, InitialU )

  BoundaryT = torch.rand( size=(BoundaryCount,1), dtype=torch.float32, requires_grad=True )*10
  l = l + burger.loss( BoundaryX, BoundaryT, BoundaryU )

  MeshX = torch.rand( size=(MeshCount,1), dtype=torch.float32, requires_grad=True )*2 - 1.0
  MeshT = torch.rand( size=(MeshCount,1), dtype=torch.float32, requires_grad=True )*4
  l = l + burger.loss( MeshX, MeshT )

  print( i, l.item() )
  l.backward()
  burger.optimizer.step()

  Loss[i] = l.item()


torch.save( burger.net.state_dict(), 'burger.pt' )


PlotX = np.linspace(-1, 1, 100)
PlotT = np.linspace(0, 5, 500)

PlotT, PlotX = np.meshgrid( PlotT, PlotX )
plotshape = PlotT.shape

PlotU = burger(
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
plt.plot( PlotX[:,int(500/5.0*2)], PlotU[:,int(500/5.0*2)], label='t=2.00' )
plt.legend()
plt.show()

Loss = np.array( Loss )

plt.plot( Loss )
plt.xlabel( 'Epochs' )
plt.yscale( 'log' )
plt.ylabel( 'Loss' )
plt.show()
