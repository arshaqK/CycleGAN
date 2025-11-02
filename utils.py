# utils.py
import torch
import os
from torchvision.utils import save_image
from glob import glob

# GAN loss (LSGAN or BCE). Use LSGAN (MSE) as in many implementations.
def gan_loss(pred, target_is_real):
    if target_is_real:
        target = torch.ones_like(pred)
    else:
        target = torch.zeros_like(pred)
    return torch.mean((pred - target)**2)

def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
    torch.save(state, filename)
    # Also save latest symlink-like file
    latest = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(state, latest)

def find_latest_checkpoint(checkpoint_dir):
    latest = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(latest):
        return latest
    # fallback: pick highest epoch file
    files = glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if not files:
        return None
    files = sorted(files)
    return files[-1]

def tensor2pil(tensor):
    # expects [-1,1] tensor
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0,1)
    return tensor

def save_samples(real_A, real_B, fake_B, fake_A, out_dir, epoch, n_samples=4):
    os.makedirs(out_dir, exist_ok=True)
    # Save a grid for A->B and B->A
    gridAB = torch.cat([real_A[:n_samples], fake_B[:n_samples]], dim=0)
    gridBA = torch.cat([real_B[:n_samples], fake_A[:n_samples]], dim=0)
    save_image(tensor2pil(gridAB), os.path.join(out_dir, f'AB_epoch_{epoch:04d}.png'), nrow=n_samples)
    save_image(tensor2pil(gridBA), os.path.join(out_dir, f'BA_epoch_{epoch:04d}.png'), nrow=n_samples)
