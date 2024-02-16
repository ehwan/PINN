#! /usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

# torch.set_default_device('cuda')

# ********* Parameters ***********
# rho = 1
Nu = 1.0/100.0
# simulation domain = [0,1]x[0,1]
# time domain = [0, T]
# dirichlet boundary condition on y=1 with u=U0, v=0
# dirichlet boundary condition on y=0, x=0, x=1 with u=0, v=0

# neumann boundary condition dp/dn = 0 on every wall

U0 = 1.0

# number of points for initial condition
N_Initial = 128

# number of points for each boundary
N_Boundary = 32

# number of points for inside domain
N_Mesh = 256

# Adam Optimizer traning epochs
Epochs = 2000

# training time step
DT = 1.0

# max time
T = 100.0

# epsilon for corner points; no training on corners
X_EPSILON = 1e-3
Y_EPSILON = 1e-3

# Step-by-step Neural Network
# train Neural Network for time \in [t, t+dt]
# could make training faster, and more accurate than training for time \in [0, T]
class LidDrivenCavityNN:
  def __init__(self, nu, dt=DT):
    self.nu = nu
    self.initial_condition = self.initial_condition0

    self.net0 = None

    self.psi0 = None
    self.p0 = None

    self.dt = dt
    self.t = 0.0

    self.init_layers()

  def init_layers( self ):
    activation = torch.nn.SiLU()
    Inputs = 64
    NumLayers = 4
    self.net = torch.nn.Sequential(
      torch.nn.Linear(3, Inputs),
    )
    for i in range(NumLayers):
      self.net.append( activation )
      self.net.append( torch.nn.Linear(Inputs, Inputs) )
      
    self.net.append( activation )
    self.net.append( torch.nn.Linear(Inputs, 2) )

    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005)
    self.optimizer_lbfgs = torch.optim.LBFGS(self.net.parameters(), max_iter=Epochs, history_size=100, line_search_fn='strong_wolfe', lr=0.1 )


  def forward(self, x, y, t):
    return self.net( torch.hstack( (x, y, t) ) )

  def init_train_inputs( self, N_Initial, N_Boundary, N_Mesh ):
    self.N_Initial = N_Initial
    self.N_Boundary = N_Boundary
    self.N_Mesh = N_Mesh

    self.X_Initial = torch.rand( size=(N_Initial,1), dtype=torch.float32, requires_grad=True )
    self.Y_Initial = torch.rand( size=(N_Initial,1), dtype=torch.float32, requires_grad=True ) * (1.0-Y_EPSILON)
    self.T_Initial = torch.zeros( size=(N_Initial,1), dtype=torch.float32, requires_grad=True )
    self.psi0, self.p0 = self.initial_condition( self.X_Initial, self.Y_Initial )

    self.X_Upper = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True ) * (1.0 - 2*X_EPSILON) + X_EPSILON
    self.Y_Upper = torch.ones( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
    self.T_Upper = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )*self.dt

    self.X_Down = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
    self.Y_Down = torch.zeros( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
    self.T_Down = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )*self.dt

    self.X_Left = torch.zeros( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
    self.Y_Left = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )*(1.0 - Y_EPSILON)
    self.T_Left = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )*self.dt

    self.X_Right = torch.ones( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )
    self.Y_Right = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )*(1.0-Y_EPSILON)
    self.T_Right = torch.rand( size=(N_Boundary,1), dtype=torch.float32, requires_grad=True )*self.dt

    self.X_Mesh = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )
    self.Y_Mesh = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )
    self.T_Mesh = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )*self.dt


  # ( x, y ) -> ( u, v, p )
  # x \in (0, 1)
  # y \in (0, 1)
  # t = 0
  def initial_condition0( self, x, y ):
    zeros = torch.zeros_like( x, dtype=torch.float32, requires_grad=False )
    return zeros, zeros

  def initial_condition1( self, x, y ):
    t = torch.ones_like( x, dtype=torch.float32, requires_grad=False )*self.prev_dt
    res = self.net0( torch.hstack( (x, y, t) ) )
    p = res[:,0:1]
    psi = res[:,1:2]
    return psi, p

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

    return u, v, p, fx, fy, psi, px, py

  def closure( self ):
    # PINN Loss

    # NS Equation on each axis
    # Incompressible Condition div u = 0
    # Initial Condition
    # Boundary Condition

    PINN_Coefficient = 1.0
    self.optimizer.zero_grad()
    l = 0

    # Initial Condition
    u, v, p, fx, fy, psi, px, py = self.function( self.X_Initial, self.Y_Initial, self.T_Initial )
    l = l + torch.mean( (psi-self.psi0)**2 + (p-self.p0)**2 + PINN_Coefficient*(fx**2 + fy**2) )
    # l = l + (fx.abs().max() + fy.abs().max() + divu.abs().max())*L_Infinity

    # Upper Boundary
    u, v, p, fx, fy, psi, px, py = self.function( self.X_Upper, self.Y_Upper, self.T_Upper )
    l = l + torch.mean( (u-U0)**2 + v**2 + py**2 + PINN_Coefficient*(fx**2 + fy**2) )

    # Down Boundary
    u, v, p, fx, fy, psi, px, py = self.function( self.X_Down, self.Y_Down, self.T_Down )
    l = l + torch.mean( u**2 + v**2 + py**2 + PINN_Coefficient*(fx**2 + fy**2) )

    # Left Boundary
    u, v, p, fx, fy, psi, px, py = self.function( self.X_Left, self.Y_Left, self.T_Left )
    l = l + torch.mean( u**2 + v**2 + px**2 + PINN_Coefficient*(fx**2 + fy**2) )

    # Right Boundary
    u, v, p, fx, fy, psi, px, py = self.function( self.X_Right, self.Y_Right, self.T_Right )
    l = l + torch.mean( u**2 + v**2 + px**2 + PINN_Coefficient*(fx**2 + fy**2) )

    # Mesh
    u, v, p, fx, fy, psi, *_ = self.function( self.X_Mesh, self.Y_Mesh, self.T_Mesh )
    l = l + torch.mean( PINN_Coefficient*(fx**2 + fy**2) )

    self.l = l
    self.l.backward( retain_graph=True )
    return self.l

  def set_dt( self, dt ):
    self.dt = dt

  # prepare new Neural Network for time \in [t + dt, t + 2*dt]
  # reserve current Neural Network for initial condition
  def step( self ):
    self.prev_dt = self.dt
    self.t = self.t + self.dt
    self.net0 = copy.deepcopy(self.net)
    self.init_layers()
    self.initial_condition = self.initial_condition1


cavity = LidDrivenCavityNN(Nu,0.1)

def iterate( Epochs ):
  Loss = [0]*Epochs
  for i in range(Epochs):
    cavity.init_train_inputs( N_Initial, N_Boundary, N_Mesh )
    l = cavity.closure()
    cavity.optimizer.step()
    print( cavity.t, i, l.item() )
    Loss[i] = l.item()


  return Loss

def save_plots( dirname ):
  PlotN = 25
  PlotX = np.linspace(0, 1, PlotN)
  PlotY = np.linspace(0, 1, PlotN)
  PlotX, PlotY = np.meshgrid( PlotX, PlotY )
  PlotT = np.ones_like(PlotX)*cavity.dt
  plotshape = PlotX.shape

  u, v, p, fx, fy, psi, *_ = cavity.function(
    torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32, requires_grad=True),
    torch.tensor(PlotY.reshape(-1,1), dtype=torch.float32, requires_grad=True),
    torch.tensor(PlotT.reshape(-1,1), dtype=torch.float32, requires_grad=True)
  )
  u = u.detach().numpy().reshape( plotshape )
  v = v.detach().numpy().reshape( plotshape )
  p = p.detach().numpy().reshape( plotshape )
  fx = fx.detach().numpy().reshape( plotshape )
  fy = fy.detach().numpy().reshape( plotshape )
  psi = psi.detach().numpy().reshape( plotshape )
  plt.quiver(
    PlotX, PlotY, u, v, np.sqrt(u**2+v**2),
  )

  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.title( 'u' )
  plt.colorbar()
  plt.savefig( dirname+'/quiver.png' )
  plt.clf()
  plt.cla()

  PlotN = 50
  PlotX = np.linspace(0, 1, PlotN)
  PlotY = np.linspace(0, 1, PlotN)
  PlotX, PlotY = np.meshgrid( PlotX, PlotY )
  PlotT = np.ones_like(PlotX)*cavity.dt
  plotshape = PlotX.shape

  u, v, p, fx, fy, psi, *_ = cavity.function(
    torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32, requires_grad=True),
    torch.tensor(PlotY.reshape(-1,1), dtype=torch.float32, requires_grad=True),
    torch.tensor(PlotT.reshape(-1,1), dtype=torch.float32, requires_grad=True)
  )
  u = u.detach().numpy().reshape( plotshape )
  v = v.detach().numpy().reshape( plotshape )
  p = p.detach().numpy().reshape( plotshape )
  fx = fx.detach().numpy().reshape( plotshape )
  fy = fy.detach().numpy().reshape( plotshape )
  psi = psi.detach().numpy().reshape( plotshape )

  plt.imshow( p, origin='lower', extent=(0,1,0,1) )
  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.colorbar()
  plt.title( 'pressure' )
  plt.savefig( dirname+'/pressure.png' )
  plt.clf()
  plt.cla()

  plt.imshow( fx, origin='lower', extent=(0,1,0,1) )
  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.colorbar()
  plt.title( 'fx' )
  plt.savefig( dirname+'/fx.png' )
  plt.clf()
  plt.cla()

  plt.imshow( fy, origin='lower', extent=(0,1,0,1) )
  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.colorbar()
  plt.title( 'fy' )
  plt.savefig( dirname+'/fy.png' )
  plt.clf()
  plt.cla()

  plt.imshow( psi, origin='lower', extent=(0,1,0,1) )
  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.colorbar()
  plt.title( 'psi' )
  plt.savefig( dirname+'/psi.png' )
  plt.clf()
  plt.cla()

  Ys = np.linspace(0, 1, 100)
  Xs = np.ones_like(Ys)*0.5
  Ts = np.ones_like(Ys)*cavity.dt

  u, v, *_ = cavity.function(
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
  plt.savefig( dirname+'/ghia.png' )
  plt.clf()
  plt.cla()

  plt.plot( Loss )
  plt.xlabel( 'Epochs' )
  plt.yscale( 'log' )
  plt.ylabel( 'Loss' )
  plt.savefig( dirname+'/loss.png' )
  plt.clf()
  plt.cla()

  with open( dirname+'/log.txt', 'w' ) as f:
    f.write( 'dt: {}\n'.format(cavity.dt) )
    f.write( 't: {}\n'.format(cavity.t+cavity.dt) )

it = 0
while cavity.t < T:
  global Loss
  Loss = iterate(Epochs)
  try:
    os.mkdir( 'results' )
  except:
    pass
  dirname = 'results/{:03d}'.format(it)
  it = it + 1
  try:
    os.mkdir( dirname )
  except:
    pass

  save_plots( dirname )

  print( 'Optimizing with L-BFGS' )
  cavity.optimizer = cavity.optimizer_lbfgs
  cavity.init_train_inputs( N_Initial, N_Boundary, N_Mesh )
  l = cavity.optimizer.step( closure=cavity.closure ).item()
  print( 'Done; Loss: {}'.format(l) )

  try:
    os.mkdir( dirname+'_lbfgs' )
  except:
    pass
  save_plots( dirname+'_lbfgs' )

  torch.save( cavity.net.state_dict(), dirname+'/cavity.pt' )
  cavity.step()
  cavity.set_dt( min(cavity.dt*2.0,1.0) )


