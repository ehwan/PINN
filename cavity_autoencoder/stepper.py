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
print( 'shape(re100raw): ', re100raw.shape )
encoder = autoencoder.CavityAutoEncoder()
encoder.load_state_dict( torch.load( 're100ae.pt' ) )

latents = encoder.encoder( re100raw )
print( 'shape(latents): ', latents.shape )

Epochs = 100
BatchSize = 20

stepper = CavityLatentStepper()

optimizer = torch.optim.Adam( stepper.parameters(), lr=0.002 )
for epoch in range(Epochs):
  print( 'Epoch: {}'.format(epoch) )
  shuffled_indices = torch.randperm( latents.shape[0]-1 )
  for batch in range(4):
    print( 'batch: {}'.format(batch) )
    batch_indices = shuffled_indices[batch*BatchSize:(batch+1)*BatchSize]
    print( 'indices: ', batch_indices )
    input_latents = latents[batch_indices].clone().detach()
    output_latents = latents[batch_indices+1].clone().detach()
    print( 'shape(input_latents): ', input_latents.shape )
    print( 'shape(output_latents): ', output_latents.shape )

    predict_output = stepper( input_latents )
    print( 'shape(predict_output): ', predict_output.shape )
    loss = torch.nn.functional.mse_loss( predict_output, output_latents )
    print( loss.item() )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save( stepper.state_dict(), 're100step.pt' )