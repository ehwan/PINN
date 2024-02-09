#! /usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt

class LidDrivenCavityNN:
  def __init__(self, nu):
    self.nu = nu
    # activation = torch.nn.Tanh()
    activation = torch.nn.SiLU()
    # activation = SinActivation()
    Inputs = 20
    NumLayers = 8
    self.net = torch.nn.Sequential(
      torch.nn.Linear(3, 64),
      activation,
      torch.nn.Linear(64,Inputs),
    )
    for i in range(NumLayers):
      self.net.append( activation )
      self.net.append( torch.nn.Linear(Inputs, Inputs) )
      
    self.net.append( activation )
    self.net.append( torch.nn.Linear(Inputs, 2) )

    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005)
    # self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=0.003)

  def forward(self, x, y, t):
    return self.net( torch.hstack( (x, y, t) ) )

  def function( self, x, y, t ):
    res = self.net( torch.hstack( (x, y, t) ) )
    p = res[:,0:1]
    psi = res[:,1:2]

    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    uy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(uy), create_graph=True)[0]
    ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    vx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    vxx = torch.autograd.grad(vx, x, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
    vy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    vyy = torch.autograd.grad(vy, y, grad_outputs=torch.ones_like(vy), create_graph=True)[0]
    vt = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    px = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    py = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    fx = ut + u*ux + v*uy + px - self.nu*(uxx + uyy)
    fy = vt + u*vx + v*vy + py - self.nu*(vxx + vyy)

    return u, v, p, fx, fy, psi


Re = 100
# simulation domain = [0,1]x[0,1]
# training time = [0, T]
# dirichlet boundary condition on y=1 with u=1, v=0
# dirichlet boundary condition on y=0, x=0, x=1 with u=0, v=0
T = 5.0
cavity = LidDrivenCavityNN(1.0/Re)
try:
  cavity.net.load_state_dict( torch.load('cavity.pt') )
except FileNotFoundError:
  pass

# number of points for initial condition
N_Initial = 64

# number of points for upper boundary
N_Upper = 64

# number of points for each boundary
N_Boundary = 16

# number of points for inside domain
N_Mesh = 128

T_Initial = torch.zeros( size=(N_Initial,1), dtype=torch.float32, requires_grad=True )
X_Initial = torch.rand( size=(N_Initial,1), dtype=torch.float32, requires_grad=True )
Y_Initial = torch.rand( size=(N_Initial,1), dtype=torch.float32, requires_grad=True )*(1.0-1e-6)

# generate random points for t=0
def InitialCondition():
  global X_Initial
  global Y_Initial
  X_Initial = torch.rand( size=(N_Initial,1), dtype=torch.float32, requires_grad=True )
  Y_Initial = torch.rand( size=(N_Initial,1), dtype=torch.float32, requires_grad=True )*(1.0-1e-6)


X_Boundary_Left = torch.zeros( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
Y_Boundary_Left = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )

X_Boundary_Right = torch.ones( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
Y_Boundary_Right = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )

X_Boundary_Down = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
Y_Boundary_Down = torch.zeros( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )

T_Boundary = torch.rand( size=(N_Boundary*3,1), dtype=torch.float32, requires_grad=True )*T

# generate random points for boundary condition
def BoundaryCondition():
  global Y_Boundary_Left
  Y_Boundary_Left = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )

  global Y_Boundary_Right
  Y_Boundary_Right = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
  
  global X_Boundary_Down
  X_Boundary_Down = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )

  global T_Boundary
  T_Boundary = torch.rand( size=(N_Boundary*3,1), dtype=torch.float32, requires_grad=True )*T

X_Upper = torch.rand( size=(N_Upper,1), dtype=torch.float32, requires_grad=True )
Y_Upper = torch.ones( size=(N_Upper,1), dtype=torch.float32, requires_grad=True )
T_Upper = torch.rand( size=(N_Upper,1), dtype=torch.float32, requires_grad=True )*T

# generate random points for upper boundary condition
def UpperBoundaryCondition():
  global X_Upper
  global T_Upper
  X_Upper = torch.rand( size=(N_Upper,1), dtype=torch.float32, requires_grad=True )
  T_Upper = torch.rand( size=(N_Upper,1), dtype=torch.float32, requires_grad=True )*T

X_Domain = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )
Y_Domain = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )
T_Domain = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )*T

# generate random points for inside domain condition ( NS equation )
def DomainCondition():
  global X_Domain
  global Y_Domain
  global T_Domain
  X_Domain = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )
  Y_Domain = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )
  T_Domain = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )*T


Epochs = 1000

Loss = [0]*Epochs

for i in range(Epochs):
  l = 0
  cavity.optimizer.zero_grad()

  # initial condition
  InitialCondition()
  u, v, _, fx, fy, _ = cavity.function( X_Initial, Y_Initial, T_Initial )
  l = l + torch.mean( u**2 + v**2 + fx**2 + fy**2 )

  # left, right, down boundary
  BoundaryCondition()
  X_Boundary = torch.vstack( (X_Boundary_Left, X_Boundary_Right, X_Boundary_Down) )
  Y_Boundary = torch.vstack( (Y_Boundary_Left, Y_Boundary_Right, Y_Boundary_Down) )
  u, v, _, fx, fy, _ = cavity.function( X_Boundary, Y_Boundary, T_Boundary )
  l = l + torch.mean( u**2 + v**2 + fx**2 + fy**2 )

  # up boundary
  UpperBoundaryCondition()
  u, v, _, fx, fy, _ = cavity.function( X_Upper, Y_Upper, T_Upper )
  l = l + torch.mean( (u-1.0)**2 + v**2 + fx**2 + fy**2 )

  # inside domain
  DomainCondition()
  u, v, _, fx, fy, _ = cavity.function( X_Domain, Y_Domain, T_Domain )
  l = l + 2*torch.mean( fx**2 + fy**2 )

  print( i, l.item() )
  l.backward()
  cavity.optimizer.step()

  Loss[i] = l.item()

torch.save( cavity.net.state_dict(), 'cavity.pt' )

PlotN = 50
PlotX = np.linspace(0, 1, PlotN)
PlotY = np.linspace(0, 1, PlotN)
PlotX, PlotY = np.meshgrid( PlotX, PlotY )
PlotT = np.ones_like( PlotX )*0.8*T
plotshape = PlotT.shape

u, v, p, fx, fy, psi = cavity.function(
  torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32, requires_grad=True),
  torch.tensor(PlotY.reshape(-1,1), dtype=torch.float32, requires_grad=True),
  torch.tensor(PlotT.reshape(-1,1), dtype=torch.float32, requires_grad=True)
)
u = u.detach().numpy().reshape( plotshape )
v = v.detach().numpy().reshape( plotshape )
p = p.detach().numpy().reshape( plotshape )
psi = psi.detach().numpy().reshape( plotshape )
fx = fx.detach().numpy().reshape( plotshape )
fy = fy.detach().numpy().reshape( plotshape )
plt.quiver(
  PlotX, PlotY, u, v, np.sqrt(u**2+v**2),
)

plt.xlabel( 'x' )
plt.ylabel( 'y' )
plt.title( 'u' )
plt.show()

plt.imshow( psi, origin='lower' )
plt.xlabel( 'x' )
plt.ylabel( 'y' )
plt.colorbar()
plt.title( 'psi' )
plt.show()

plt.imshow( p, origin='lower' )
plt.xlabel( 'x' )
plt.ylabel( 'y' )
plt.colorbar()
plt.title( 'pressure' )
plt.show()

plt.imshow( fx, origin='lower' )
plt.xlabel( 'x' )
plt.ylabel( 'y' )
plt.colorbar()
plt.title( 'fx' )
plt.show()

plt.imshow( fy, origin='lower' )
plt.xlabel( 'x' )
plt.ylabel( 'y' )
plt.colorbar()
plt.title( 'fy' )
plt.show()

Ys = np.linspace(0, 1, 100)
Xs = np.ones_like(Ys)*0.5
Ts = np.ones_like( Xs )*0.8*T

u, v, p, fx, fy, psi = cavity.function(
  torch.tensor(Xs.reshape(-1,1), dtype=torch.float32, requires_grad=True),
  torch.tensor(Ys.reshape(-1,1), dtype=torch.float32, requires_grad=True),
  torch.tensor(Ts.reshape(-1,1), dtype=torch.float32, requires_grad=True)
)
u = u.detach().numpy().reshape( -1 )
plt.plot( u, Ys, label='Neural Network' )


# Ghia et al. (1982)
Ys = [
  1.00000
  ,0.9766
  ,0.9688
  ,0.9609
  ,0.9531
  ,0.8516
  ,0.7344
  ,0.6172
  ,0.5000
  ,0.4531
  ,0.2813
  ,0.1719
  ,0.1016
  ,0.0703
  ,0.0625
  ,0.0547
  ,0.0000
]

Us =[
  1.00000
  ,0.84123
  ,0.78871
  ,0.73722
  ,0.68717
  ,0.23151
  ,0.00332
  ,-0.13641
  ,-0.20581
  ,-0.21090
  ,-0.15662
  ,-0.10150
  ,-0.06434
  ,-0.04775
  ,-0.04192
  ,-0.03717
  ,0.00000
]

plt.plot( Us, Ys, 'o-', label='Ghia et al.' )
plt.xlabel( 'u' )
plt.ylabel( 'y' )
plt.legend()
plt.show()

Loss = np.array( Loss )

plt.plot( Loss )
plt.xlabel( 'Epochs' )
plt.yscale( 'log' )
plt.ylabel( 'Loss' )
plt.show()
