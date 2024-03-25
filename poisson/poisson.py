#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float32)


class PoissonEquationNN(torch.nn.Module):
  def __init__(self, activation, input_dim, output_dim, hidden_dim, hidden_layers=4):
    super(PoissonEquationNN, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      activation
    )
    for i in range(hidden_layers):
      self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
      self.net.append(activation)

    self.net.append( torch.nn.Linear(hidden_dim, output_dim) )

  def set_dirichlet(self, inputs, dirichlets):
    self.dirichlet_inputs = []
    for i in range(self.input_dim):
      x = inputs[i].reshape(-1,1).detach().clone()
      x.requires_grad = False
      self.dirichlet_inputs.append( x )

    self.dirichlet_outputs = dirichlets.reshape(-1,self.output_dim).detach().clone()
    self.dirichlet_outputs.requires_grad = False
  
  def set_input( self, inputs, poisson_rhs ):
    self.inputs = []
    for i in range(self.input_dim):
      x = inputs[i].reshape(-1,1).detach().clone()
      x.requires_grad = True
      self.inputs.append( x )

    self.poisson_rhs = poisson_rhs.reshape(-1,self.output_dim).detach().clone()

  def set_neumann(self, inputs, normals, neumanns):
    self.neumann_inputs = []
    for i in range(self.input_dim):
      x = inputs[i].reshape(-1,1).detach().clone()
      x.requires_grad = True
      self.neumann_inputs.append( x )

    self.neumann_normals = normals.reshape(-1,self.input_dim).detach().clone()
    self.neumann_normals.requires_grad = False
    self.neumann_outputs = neumanns.reshape(-1,self.output_dim).detach().clone()
    self.neumann_outputs.requires_grad = False

  def loss( self ):
    l = 0

    # PINN loss
    phi = self.net(torch.hstack(self.inputs))
    lap_phi = 0
    for x in self.inputs:
      phi_x = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True)[0]
      phi_xx = torch.autograd.grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0]
      lap_phi = lap_phi + phi_xx

    l = l + torch.mean( (lap_phi - self.poisson_rhs)**2 )

    # Dirichlet loss
    if self.dirichlet_inputs is not None:
      phi = self.net(torch.hstack(self.dirichlet_inputs))
      l = l + torch.mean( (phi - self.dirichlet_outputs)**2 )

    return l


  def forward(self, x):
    return self.net(x)


nn = PoissonEquationNN( torch.nn.SiLU(), 2, 1, 32, 8 )

omega = 4
N_Boundary = 128
N_Mesh = 64
PlotN = 500
Epochs = 1000
Iteration = 2000

# boundary conditions
Lins = np.linspace(0,1,N_Boundary)
Zeros = np.zeros_like(Lins)
Ones = np.ones_like(Lins)

Xs = np.hstack( (Zeros, Ones, Lins, Lins) )
Ys = np.hstack( (Lins, Lins, Zeros, Ones) )
Dirichlet = np.zeros_like(Xs)

nn.set_dirichlet( [torch.tensor(Xs,dtype=torch.float32), torch.tensor(Ys,dtype=torch.float32)], torch.tensor(Dirichlet,dtype=torch.float32) )

Lins = np.linspace(0,1,N_Mesh)
Xs, Ys = np.meshgrid(Lins, Lins)
Poisson_rhs = np.sin( np.pi*Xs ) * np.sin( np.pi*Ys ) + np.sin( 2*np.pi*Xs ) * np.sin( 2*np.pi*Ys ) + np.sin( 4*np.pi*Xs ) * np.sin( 4*np.pi*Ys )
nn.set_input( [torch.tensor(Xs,dtype=torch.float32), torch.tensor(Ys,dtype=torch.float32)], torch.tensor(Poisson_rhs,dtype=torch.float32) )

adam = torch.optim.Adam(nn.parameters(), lr=1e-2)

loss = []
for i in range(Epochs):
  adam.zero_grad()
  l = nn.loss()
  l.backward()
  adam.step()
  loss.append( l.item() )
  if i % 100 == 0:
    print( i, l.item() )

lbfgs = torch.optim.LBFGS(nn.parameters(), lr=1e-1, max_iter=Iteration, line_search_fn='strong_wolfe')
def closure():
  lbfgs.zero_grad()
  l = nn.loss()
  l.backward()
  loss.append( l.item() )
  print( l.item() )
  return l
lbfgs.step(closure)


plt.plot( loss )
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

Xs = np.linspace(0,1,PlotN)
Ys = np.linspace(0,1,PlotN)
Xs, Ys = np.meshgrid(Xs, Ys)
PlotShape = Xs.shape
Xs = torch.tensor( Xs.reshape(-1,1), dtype=torch.float32 )
Ys = torch.tensor( Ys.reshape(-1,1), dtype=torch.float32 )
Zs = nn( torch.hstack( (Xs,Ys) ) ).detach().numpy().reshape( PlotShape )
plt.imshow( Zs, origin='lower' )
plt.xlabel( 'x' )
plt.ylabel( 'y' )
plt.colorbar()
plt.title( 'psi' )
plt.show()