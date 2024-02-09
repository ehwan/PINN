#!/usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt

class SpringEquationNN(torch.nn.Module):
  def __init__(self, k, mu):
    super().__init__()
    self.k = k
    self.mu = mu
    activation = torch.nn.SiLU()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(1,20),
      activation,
      torch.nn.Linear(20,20),
      activation,
      torch.nn.Linear(20,20),
      activation,
      torch.nn.Linear(20,20),
      activation,
      torch.nn.Linear(20,20),
      activation,
      torch.nn.Linear(20,20),
      activation,
      torch.nn.Linear(20,20),
      activation,
      torch.nn.Linear(20,1)
    )

    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005)

  def forward(self, t):
    return self.net( t )

  def loss( self, t, x_answer=None, xt_answer=None ):
    x = self.net( t )
    xt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    xtt = torch.autograd.grad(xt, t, grad_outputs=torch.ones_like(xt), create_graph=True)[0]
    f = xtt + self.mu * xt + self.k * x
    loss = torch.mean( f**2 )
    if x_answer is not None:
      loss = loss + torch.mean( (x - x_answer)**2 )
    if xt_answer is not None:
      loss = loss + torch.mean( (xt - xt_answer)**2 )
    return loss


K = 1.0
Mu = 1.02
spring = SpringEquationNN(K, Mu)

Epochs = 3000
BatchSize = 256
T = 30.0
X0 = 1.0
V0 = -1.0

Loss = [0]*Epochs

for i in range(Epochs):
  t0 = torch.zeros( (BatchSize,1), dtype=torch.float32, requires_grad=True )
  x0 = torch.ones_like( t0, dtype=torch.float32, requires_grad=False )*X0
  xt0 = torch.ones_like( t0, dtype=torch.float32, requires_grad=False )*V0
  t = torch.rand( (BatchSize,1), dtype=torch.float32, requires_grad=True )*T

  spring.optimizer.zero_grad()
  loss = 0
  loss = loss + spring.loss( t0, x0, xt0 )
  loss = loss + spring.loss( t )
  loss.backward()
  spring.optimizer.step()

  Loss[i] = loss.item()
  print( i, loss.item() )



Ts = torch.linspace(0, 30, 1000).reshape(-1,1)
Xs = spring.net( Ts )

Ts = Ts.detach().numpy().reshape(-1)
Xs = Xs.detach().numpy().reshape(-1)

plt.plot( Ts, Xs )

# 4th order Runge-Kutta
Y = np.array( [X0, V0] )
runge_Xs = np.zeros_like( Ts )
runge_Xs[0] = Y[0]
h = Ts[1]- Ts[0]
for i, t in enumerate(Ts[:-1]):
  def f( t, y ):
    return np.array( [y[1], -Mu*y[1] - K*y[0]] )
  k1 = f(t, Y)
  k2 = f(t + h/2, Y + h/2*k1)
  k3 = f(t + h/2, Y + h/2*k2)
  k4 = f(t + h, Y + h*k3)
  Y = Y + h/6*(k1 + 2*k2 + 2*k3 + k4)
  runge_Xs[i+1] = Y[0]

plt.plot( Ts, runge_Xs )

plt.xlabel( 't' )
plt.ylabel( 'x' )
plt.show()


plt.plot( Loss )
plt.xlabel( 'Epoch' ) 
plt.ylabel( 'Loss' )
plt.yscale('log')
plt.show()

plt.plot( Ts, (Xs-runge_Xs)**2 )
plt.xlabel( 't' )
plt.ylabel( 'Error' )
plt.show()