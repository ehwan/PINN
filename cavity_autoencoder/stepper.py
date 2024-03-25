import torch
import autoencoder
import data_loader

class CavityLatentStepper(torch.nn.Module):
  def __init__(self):
    super(CavityLatentStepper, self).__init__()

    self.step = torch.nn.Sequential(
      torch.nn.Linear( 32, 128 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 128, 128),
      torch.nn.SiLU(),
      torch.nn.Linear( 128, 128 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 128, 32 )
    )

  def forward( self, latent ):
    return self.step( latent )


re100raw = data_loader.load_file( 'trains/re100.dat' )
encoder = autoencoder.CavityAutoEncoder()
encoder.load_state_dict( torch.load( 're100ae.pt' ) )

latents = encoder.encoder( re100raw )

Epochs = 100
BatchSize = 20

stepper = CavityLatentStepper()

optimizer = torch.optim.Adam( stepper.parameters(), lr=0.002 )
for epoch in range(Epochs):
  print( 'Epoch: {}'.format(epoch) )
  for batch in range(4):
    print( 'batch: {}'.format(batch) )
    input_latents = latents[batch*BatchSize:(batch+1)*BatchSize]
    output_latents = latents[batch*BatchSize+1:(batch+1)*BatchSize+1]

    predict_output = stepper( input_latents )
    loss = torch.nn.functional.mse_loss( predict_output, output_latents )
    print( loss.item() )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save( stepper.state_dict(), 're100step.pt' )