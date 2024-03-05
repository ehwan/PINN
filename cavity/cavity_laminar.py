#! /usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

# ********* Parameters ***********
# rho = 1
Nu = 1.0/100.0
# simulation domain = [0,1]x[0,1]
# time domain = [0, T]
# dirichlet boundary condition on y=1 with u=U0, v=0
# dirichlet boundary condition on y=0, x=0, x=1 with u=0, v=0

# neumann boundary condition dp/dn = 0 on every wall
U0 = 1.0

# number of points for inside domain
N_Mesh = 512

# Adam Optimizer traning epochs
Epochs = 2000

# L-BFGS Optimizer max iterations
MaxIter = 5000

class ABlock:
  def __init__(self, activation):
    self.activation = activation
    N = 16
    self.net = torch.nn.Sequential(
      torch.nn.Linear(2,N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, 1),
    )
    Batch = 64
    Lins = np.linspace(0,1,Batch)
    Ones = np.ones( Batch )
    Zeros = np.zeros( Batch )
    self.X_Left = torch.tensor( Zeros, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Left = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Right = torch.tensor( Ones, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Right = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Bottom = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Bottom = torch.tensor( Zeros, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Top = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Top = torch.tensor( Ones, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_half = torch.tensor( [0.5], dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_half = torch.tensor( [0.5], dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )

  def forward( self, input ):
    return self.net(input)

  def function( self, x, y ):
    res = self.forward( torch.hstack((x,y)) )

    res_x = torch.autograd.grad(res, x, grad_outputs=torch.ones_like(res), create_graph=True)[0]
    res_y = torch.autograd.grad(res, y, grad_outputs=torch.ones_like(res), create_graph=True)[0]
    return res, res_x, res_y

  def loss( self ):
    l = 0

    res, res_x, res_y = self.function( self.X_Left, self.Y_Left )
    l = l + torch.mean( res**2 + res_x**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_Right, self.Y_Right )
    l = l + torch.mean( res**2 + res_x**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_Bottom, self.Y_Bottom )
    l = l + torch.mean( res**2 + res_x**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_Top, self.Y_Top )
    l = l + torch.mean( res**2 + res_x**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_half, self.Y_half )
    l = l + torch.mean( (res-1)**2 )

    return l

  def train( self ):
    adam = torch.optim.Adam( self.net.parameters(), lr=0.005 )
    print( 'training ABlock' )
    for i in range(10000):
      l = self.loss()
      adam.zero_grad()
      l.backward()
      adam.step()
      if i % 1000 == 0:
        print( i, l.item() )
class BBlock:
  def __init__(self, activation):
    self.activation = activation
    N = 16
    self.net = torch.nn.Sequential(
      torch.nn.Linear(2,N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, 1),
    )
    Batch = 64
    Lins = np.linspace(0,1,Batch)
    Ones = np.ones( Batch )
    Zeros = np.zeros( Batch )
    self.X_Left = torch.tensor( Zeros, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Left = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Right = torch.tensor( Ones, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Right = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Bottom = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Bottom = torch.tensor( Zeros, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Top = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Top = torch.tensor( Ones, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_half = torch.tensor( [0.5], dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_half = torch.tensor( [0.5], dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )

  def forward( self, input ):
    return self.net(input)

  def function( self, x, y ):
    res = self.forward( torch.hstack((x,y)) )

    res_x = torch.autograd.grad(res, x, grad_outputs=torch.ones_like(res), create_graph=True)[0]
    res_y = torch.autograd.grad(res, y, grad_outputs=torch.ones_like(res), create_graph=True)[0]
    return res, res_x, res_y

  def loss( self ):
    l = 0

    res, res_x, res_y = self.function( self.X_Left, self.Y_Left )
    l = l + torch.mean( res_x**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_Right, self.Y_Right )
    l = l + torch.mean( res_x**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_Bottom, self.Y_Bottom )
    l = l + torch.mean( res_x**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_Top, self.Y_Top )
    l = l + torch.mean( res_x**2 + (res_y-1)**2 )
    res, res_x, res_y = self.function( self.X_half, self.Y_half )
    l = l + torch.mean( (res-1)**2 )

    return l

  def train( self ):
    adam = torch.optim.Adam( self.net.parameters(), lr=0.005 )
    print( 'training BBlock' )
    for i in range(10000):
      l = self.loss()
      adam.zero_grad()
      l.backward()
      adam.step()
      if i % 1000 == 0:
        print( i, l.item() )

class CBlock:
  def __init__(self, activation):
    self.activation = activation
    N = 16
    self.net = torch.nn.Sequential(
      torch.nn.Linear(2,N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, N),
      self.activation,
      torch.nn.Linear(N, 1),
    )
    Batch = 64
    Lins = np.linspace(0,1,Batch)
    Ones = np.ones( Batch )
    Zeros = np.zeros( Batch )
    self.X_Left = torch.tensor( Zeros, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Left = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Right = torch.tensor( Ones, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Right = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Bottom = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Bottom = torch.tensor( Zeros, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_Top = torch.tensor( Lins, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_Top = torch.tensor( Ones, dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.X_half = torch.tensor( [0.5], dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )
    self.Y_half = torch.tensor( [0.5], dtype=torch.float32, requires_grad=True ).reshape( -1, 1 )

  def forward( self, input ):
    return self.net(input)

  def function( self, x, y ):
    res = self.forward( torch.hstack((x,y)) )

    res_x = torch.autograd.grad(res, x, grad_outputs=torch.ones_like(res), create_graph=True)[0]
    res_y = torch.autograd.grad(res, y, grad_outputs=torch.ones_like(res), create_graph=True)[0]
    return res, res_x, res_y

  def loss( self ):
    l = 0

    res, res_x, res_y = self.function( self.X_Left, self.Y_Left )
    l = l + torch.mean( res**2 + res_x**2 )
    res, res_x, res_y = self.function( self.X_Right, self.Y_Right )
    l = l + torch.mean( res**2 + res_x**2 )
    res, res_x, res_y = self.function( self.X_Bottom, self.Y_Bottom )
    l = l + torch.mean( res**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_Top, self.Y_Top )
    l = l + torch.mean( res**2 + res_y**2 )
    res, res_x, res_y = self.function( self.X_half, self.Y_half )
    l = l + torch.mean( (res-1)**2 )

    return l

  def train( self ):
    adam = torch.optim.Adam( self.net.parameters(), lr=0.005 )
    print( 'training CBlock' )
    for i in range(10000):
      l = self.loss()
      adam.zero_grad()
      l.backward()
      adam.step()
      if i % 1000 == 0:
        print( i, l.item() )




class ResidualBlock(torch.nn.Module):
  def __init__(self, activation, inout, middle):
    super(ResidualBlock, self).__init__()
    self.activation = activation
    self.net = torch.nn.Sequential(
      torch.nn.Linear(inout, middle),
      self.activation,
      torch.nn.Linear(middle, inout),
    )

  def forward( self, input ):
    return input + self.net(input)

class LidDrivenCavityNN(torch.nn.Module):
  def __init__(self, nu):
    super(LidDrivenCavityNN, self).__init__()
    self.nu = nu
    self.init_layers()
    self.ablock = ABlock( torch.nn.SiLU() )
    self.ablock.train()
    self.bblock = BBlock( torch.nn.SiLU() )
    self.bblock.train()
    self.cblock = CBlock( torch.nn.SiLU() )
    self.cblock.train()

  def init_layers( self ):
    activation = torch.nn.SiLU()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(2, 64),
      activation,
      torch.nn.Linear(64,64),
      activation,
      torch.nn.Linear(64,64),
      activation,
      torch.nn.Linear(64,64),
      activation,
      torch.nn.Linear(64,64),
      activation,
      torch.nn.Linear(64,64),
      activation,
      torch.nn.Linear(64,2)
    )

  def init_train_inputs( self, N_Mesh ):
    Xs = np.linspace(0, 1, 32)
    Ys = np.linspace(0, 1, 32)
    Xs, Ys = np.meshgrid( Xs, Ys )
    self.X_Mesh = torch.tensor( Xs.reshape(-1,1), dtype=torch.float32, requires_grad=True )
    self.Y_Mesh = torch.tensor( Ys.reshape(-1,1), dtype=torch.float32, requires_grad=True )

    # self.N_Mesh = N_Mesh

    # self.X_Mesh = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )
    # self.Y_Mesh = torch.rand( size=(N_Mesh,1), dtype=torch.float32, requires_grad=True )

  def forward( self, input ):
    return self.net( input )

  def function( self, x, y ):
    res = self.forward( torch.hstack((x,y)) )
    p = res[:,0:1]
    psi = res[:,1:2]

    A = self.ablock.forward( torch.hstack((x,y)) )
    B = self.bblock.forward( torch.hstack((x,y)) )
    C = self.cblock.forward( torch.hstack((x,y)) )

    # force boundary condition
    p = C*p
    psi = A*psi + B

    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    uy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(uy), create_graph=True)[0]

    vx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    vxx = torch.autograd.grad(vx, x, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
    vy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    vyy = torch.autograd.grad(vy, y, grad_outputs=torch.ones_like(vy), create_graph=True)[0]

    px = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    py = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    fx = u*ux + v*uy + px - self.nu*(uxx + uyy)
    fy = u*vx + v*vy + py - self.nu*(vxx + vyy)

    return u, v, p, fx, fy, psi, px, py, A, B, C

  def loss( self ):
    l = 0

    # Mesh
    u, v, p, fx, fy, psi, *_ = self.function( self.X_Mesh, self.Y_Mesh )
    l = l + torch.mean( (fx**2 + fy**2) )

    return l

loss = []

def train_with_adam( nn, epochs ):
  print( 'Training with Adam' )
  adam = torch.optim.Adam( nn.parameters(), lr=0.003 )

  loss = [0] * epochs
  nn.init_train_inputs( N_Mesh )
  for i in range(epochs):
    adam.zero_grad()
    l = nn.loss()
    l.backward()
    adam.step()
    loss[i] = l.item()
    print( 'Epoch: {}; Loss: {}'.format(i, l.item()) )

  return loss


lbfgs_iteration = 0
def train_with_lbfgs( nn, max_iter ):
  print( 'Training with L-BFGS' )
  lbfgs = torch.optim.LBFGS( nn.parameters(), lr=0.8, max_iter=max_iter, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe' )

  nn.init_train_inputs( N_Mesh )
  global lbfgs_iteration

  lbfgs_iteration = 0
  def closure():
    global lbfgs_iteration
    lbfgs.zero_grad()
    l = nn.loss()
    loss.append( l.item() )
    l.backward( retain_graph=True )
    print( 'Iteration: {}; Loss: {}'.format(lbfgs_iteration, l.item()) )
    lbfgs_iteration = lbfgs_iteration + 1
    return l

  lbfgs.step( closure )

  return loss



def save_plots( nn, dirname, loss ):
  os.makedirs( dirname, exist_ok=True )
  PlotN = 25
  PlotX = np.linspace(0, 1, PlotN)
  PlotY = np.linspace(0, 1, PlotN)
  PlotX, PlotY = np.meshgrid( PlotX, PlotY )
  plotshape = PlotX.shape

  u, v, p, fx, fy, psi, *_ = nn.function(
    torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32, requires_grad=True),
    torch.tensor(PlotY.reshape(-1,1), dtype=torch.float32, requires_grad=True)
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
  plotshape = PlotX.shape

  u, v, p, fx, fy, psi, px, py, A, B, C = nn.function(
    torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32, requires_grad=True),
    torch.tensor(PlotY.reshape(-1,1), dtype=torch.float32, requires_grad=True)
  )
  u = u.detach().numpy().reshape( plotshape )
  v = v.detach().numpy().reshape( plotshape )
  p = p.detach().numpy().reshape( plotshape )
  fx = fx.detach().numpy().reshape( plotshape )
  fy = fy.detach().numpy().reshape( plotshape )
  psi = psi.detach().numpy().reshape( plotshape )
  A = A.detach().numpy().reshape( plotshape )
  B = B.detach().numpy().reshape( plotshape )
  C = C.detach().numpy().reshape( plotshape )

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

  plt.imshow( A, origin='lower', extent=(0,1,0,1) )
  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.colorbar()
  plt.title( 'A' )
  plt.savefig( dirname+'/A.png' )
  plt.clf()
  plt.cla()

  plt.imshow( B, origin='lower', extent=(0,1,0,1) )
  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.colorbar()
  plt.title( 'B' )
  plt.savefig( dirname+'/B.png' )
  plt.clf()
  plt.cla()

  plt.imshow( C, origin='lower', extent=(0,1,0,1) )
  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.colorbar()
  plt.title( 'C' )
  plt.savefig( dirname+'/C.png' )
  plt.clf()
  plt.cla()

  Ys = np.linspace(0, 1, 100)
  Xs = np.ones_like(Ys)*0.5

  u, v, *_ = nn.function(
    torch.tensor(Xs.reshape(-1,1), dtype=torch.float32, requires_grad=True),
    torch.tensor(Ys.reshape(-1,1), dtype=torch.float32, requires_grad=True)
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

  plt.plot( loss )
  plt.xlabel( 'Epochs' )
  plt.yscale( 'log' )
  plt.ylabel( 'Loss' )
  plt.savefig( dirname+'/loss.png' )
  plt.clf()
  plt.cla()

  torch.save( nn.state_dict(), dirname+'/cavity.pt' )


cavity = LidDrivenCavityNN( Nu )
loss = train_with_adam( cavity, Epochs )
save_plots( cavity, 'results_adam', loss )
loss = train_with_lbfgs( cavity, MaxIter )
save_plots( cavity, 'results_lbfgs', loss )