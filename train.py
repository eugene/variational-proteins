import torch
import numpy as np
from misc import data, c
from torch import optim
from scipy.stats import spearmanr
from vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader, df, mutants_tensor, mutants_df = data(batch_size = 64)

wildtype   = dataloader.dataset[0] # one-hot-encoded wildtype 
eval_batch = torch.cat([wildtype.unsqueeze(0), mutants_tensor])

args = {
    'alphabet_size': dataloader.dataset[0].shape[0],
    'seq_len':       dataloader.dataset[0].shape[1]
}

vae   = VAE(**args).to(device)
opt   = optim.Adam(vae.parameters())

# rl  = Reconstruction loss
# kl  = Kullback-Leibler divergence loss
# cor = Spearman correlation to experimentally measured 
#       protein fitness according to eq.1 from paper
stats = { 'rl': [], 'kl': [], 'cor': [] }

for epoch in range(32):
    # Unsupervised training on the MSA sequences.
    vae.train()
    
    epoch_losses = { 'rl': [], 'kl': [] }
    for batch in dataloader:
        opt.zero_grad()
        x_hat, mu, logvar = vae(batch)
        loss, rl, kl      = vae.loss(x_hat, batch, mu, logvar)
        loss.mean().backward()
        opt.step()
        epoch_losses['rl'].append(rl.mean().item())
        epoch_losses['kl'].append(kl.mean().item())

    # Evaluation on mutants
    vae.eval()
    x_hat_eval, mu, logvar = vae(eval_batch, rep=False)
    elbos, _, _ = vae.loss(x_hat_eval, eval_batch, mu, logvar)
    diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
    cor, _      = spearmanr(mutants_df.value, diffs.detach())
    
    # Populate statistics 
    stats['rl'].append(np.mean(epoch_losses['rl']))
    stats['kl'].append(np.mean(epoch_losses['kl']))
    stats['cor'].append(np.abs(cor))

    to_print = [
        f"{c.HEADER}EPOCH %03d"          % epoch,
        f"{c.OKBLUE}RL=%4.4f"            % stats['rl'][-1], 
        f"{c.OKGREEN}KL=%4.4f"           % stats['kl'][-1], 
        f"{c.OKCYAN}|rho|=%4.4f{c.ENDC}" % stats['cor'][-1]
    ]
    print(" ".join(to_print))

torch.save({
    'state_dict': vae.state_dict(), 
    'stats':      stats,
    'args':       args,
}, "trained.model.pth")