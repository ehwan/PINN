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

wave = WaveEquationNN(1)

T1 = 4.0
N1 = 64
T2 = 8.0
N2 = 64

BoundaryN = 128

Epochs = 25000

Loss = [0]*Epochs

for i in range(Epochs):
  l = 0
  wave.optimizer.zero_grad()

  DataX1 = torch.rand( size=(N1,1), dtype=torch.float32, requires_grad=True )*2 - 1.0
  DataT1 = torch.rand( size=(N1,1), dtype=torch.float32, requires_grad=True )*T1
  DataU1 = torch.sin(np.pi*DataX1)*torch.cos(np.pi*DataT1)
  l = l + wave.loss( DataX1, DataT1, DataU1 )

  DataX2 = torch.rand( size=(N2,1), dtype=torch.float32, requires_grad=True )*2 - 1.0
  DataT2 = torch.rand( size=(N2,1), dtype=torch.float32, requires_grad=True )*(T2-T1) + T1
  l = l + 2*wave.loss( DataX2, DataT2 )

  BoundaryT = torch.rand( size=(BoundaryN*2,1), dtype=torch.float32, requires_grad=True )*T2
  BoundaryX = torch.ones( size=(BoundaryN,1), dtype=torch.float32, requires_grad=True )
  BoundaryX = torch.vstack( (BoundaryX, -BoundaryX) )
  BoundaryU = torch.zeros_like(BoundaryX)
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

PlotU = wave.forward(
  torch.tensor(PlotX.reshape(-1,1), dtype=torch.float32),
  torch.tensor(PlotT.reshape(-1,1), dtype=torch.float32)
)

PlotU = PlotU.detach().numpy().reshape( plotshape )

plt.imshow( PlotU, aspect='auto', extent=[0, 10, -1, 1], vmin=-1.5, vmax=1.5 )
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