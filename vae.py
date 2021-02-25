# This VAE is as vanilla as it can be.
import torch

class VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.hidden_size   = 64
        self.latent_size   = 2
        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len       = kwargs['seq_len']
        self.input_size    = self.alphabet_size * self.seq_len

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        )

        # Latent space `mu` and `var`
        self.fc21 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.fc22 = torch.nn.Linear(self.hidden_size, self.latent_size)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.input_size),
        )

    def forward(self, x, rep=True):
        x = x.view(-1, self.input_size)                    # flatten
        x = self.encoder(x)                                # encode
        mu, logvar = self.fc21(x), self.fc22(x)            # branch mu, var

        if rep:                                            # reparameterize
            x = mu + torch.randn_like(mu) * (0.5*logvar).exp() 
        else:                                              # or don't 
            x = mu                                         

        x = self.decoder(x)                                # decode
        x = x.view(-1, self.alphabet_size, self.seq_len)   # squeeze back
        x = x.log_softmax(dim=1)                           # softmax
        return x, mu, logvar
    
    def loss(self, x_hat, true_x, mu, logvar, beta=0.5):
        RL = -(x_hat*true_x).sum(-1).sum(-1)                    # reconst. loss
        KL = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1) # KL loss
        return RL + beta*KL, RL, KL