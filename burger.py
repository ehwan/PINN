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

InitialX = np.linspace(-1, 1, 100)
InitialT = np.zeros_like(InitialX)
InitialU = -np.sin(np.pi*InitialX)

BoundaryT = np.linspace(0, 10, 500)[1:]
BoundaryX = np.ones_like(BoundaryT)
BoundaryU = np.zeros_like(BoundaryT)

BoundaryX = np.concatenate( (InitialX, BoundaryX, -BoundaryX), axis=0 )
BoundaryT = np.concatenate( (InitialT, BoundaryT, BoundaryT) , axis=0)
BoundaryU = np.concatenate( (InitialU, BoundaryU, BoundaryU) , axis=0)

BoundaryX = torch.tensor( BoundaryX.reshape(-1,1), dtype=torch.float32, requires_grad=True )
BoundaryT = torch.tensor( BoundaryT.reshape(-1,1), dtype=torch.float32, requires_grad=True )
BoundaryU = torch.tensor( BoundaryU.reshape(-1,1), dtype=torch.float32, requires_grad=False )

MeshX = np.linspace(-1, 1, 100)[1:-1]
MeshT = np.linspace(0, 10, 500)[1:]
MeshX, MeshT = np.meshgrid( MeshX, MeshT )
MeshX = torch.tensor( MeshX.reshape(-1,1), dtype=torch.float32, requires_grad=True )
MeshT = torch.tensor( MeshT.reshape(-1,1), dtype=torch.float32, requires_grad=True )

Epochs = 500

Loss1 = [0] * Epochs
Loss2 = [0] * Epochs

for i in range(Epochs):
  print( i )
  Loss1[i] = burger.train( BoundaryX, BoundaryT, BoundaryU )
  Loss2[i] = burger.train( MeshX, MeshT )
  print( Loss1[i] )
  print( Loss2[i] )


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

Loss1 = np.array( Loss1 )
Loss2 = np.array( Loss2 )

plt.plot( range(Epochs), np.log(Loss1), label='Boundary' )
plt.plot( range(Epochs), np.log(Loss2), label='Mesh' )
plt.legend()
plt.show()
