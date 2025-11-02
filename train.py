# train.py
import os
import torch
from torch.utils.data import DataLoader
from models import ResnetGenerator, NLayerDiscriminator
from datasets import UnalignedDataset
import utils
import argparse
from tqdm import tqdm
import itertools
import torch.optim as optim
import torch.nn as nn

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = requires_grad

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize networks
    netG_A2B = ResnetGenerator(3,3).to(device)  # A (sketch) -> B (photo)
    netG_B2A = ResnetGenerator(3,3).to(device)  # B -> A
    netD_A = NLayerDiscriminator(3).to(device)
    netD_B = NLayerDiscriminator(3).to(device)

    # optimizers
    lr = args.lr
    optG = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
    optD_A = optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optD_B = optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # schedulers (linear decay after n_epochs_decay)
    def lambda_rule(epoch):
        if epoch < args.n_epochs:
            return 1.0
        else:
            return max(0.0, 1.0 - float(epoch - args.n_epochs) / (args.n_epochs_decay + 1))
    schedulerG = optim.lr_scheduler.LambdaLR(optG, lr_lambda=lambda_rule)
    schedulerD_A = optim.lr_scheduler.LambdaLR(optD_A, lr_lambda=lambda_rule)
    schedulerD_B = optim.lr_scheduler.LambdaLR(optD_B, lr_lambda=lambda_rule)

    # resume if checkpoint exists
    start_epoch = 1
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest = utils.find_latest_checkpoint(args.checkpoint_dir)
    if latest:
        print(f"Resuming from checkpoint {latest}")
        data = torch.load(latest, map_location=device)
        netG_A2B.load_state_dict(data['netG_A2B'])
        netG_B2A.load_state_dict(data['netG_B2A'])
        netD_A.load_state_dict(data['netD_A'])
        netD_B.load_state_dict(data['netD_B'])
        optG.load_state_dict(data['optG'])
        optD_A.load_state_dict(data['optD_A'])
        optD_B.load_state_dict(data['optD_B'])
        start_epoch = data.get('epoch', 1) + 1

    # data
    dataset = UnalignedDataset(args.dataset_root, phase='train', load_size=args.load_size, crop_size=args.crop_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # loss weights
    lambda_cycle = args.lambda_cycle
    lambda_id = args.lambda_id

    mse_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    for epoch in range(start_epoch, args.total_epochs+1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.total_epochs}")
        for i, data in enumerate(loop):
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)

            ###### Generators A2B and B2A ######
            set_requires_grad([netD_A, netD_B], False)
            optG.zero_grad()

            # Identity loss
            idt_B = netG_A2B(real_B)
            loss_idt_B = l1_loss(idt_B, real_B) * lambda_cycle * lambda_id
            idt_A = netG_B2A(real_A)
            loss_idt_A = l1_loss(idt_A, real_A) * lambda_cycle * lambda_id

            # GAN loss A->B
            fake_B = netG_A2B(real_A)
            pred_fake_B = netD_B(fake_B)
            loss_GAN_A2B = torch.mean((pred_fake_B - 1)**2)

            # GAN loss B->A
            fake_A = netG_B2A(real_B)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_B2A = torch.mean((pred_fake_A - 1)**2)

            # Cycle loss
            rec_A = netG_B2A(fake_B)
            loss_cycle_A = l1_loss(rec_A, real_A) * lambda_cycle
            rec_B = netG_A2B(fake_A)
            loss_cycle_B = l1_loss(rec_B, real_B) * lambda_cycle

            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()
            optG.step()

            ###### Discriminator A ######
            set_requires_grad([netD_A], True)
            optD_A.zero_grad()
            pred_real = netD_A(real_A)
            loss_D_real = torch.mean((pred_real - 1)**2)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = torch.mean((pred_fake - 0)**2)
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optD_A.step()

            ###### Discriminator B ######
            set_requires_grad([netD_B], True)
            optD_B.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = torch.mean((pred_real - 1)**2)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = torch.mean((pred_fake - 0)**2)
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optD_B.step()

            loop.set_postfix({
                'loss_G': loss_G.item(),
                'loss_D_A': loss_D_A.item(),
                'loss_D_B': loss_D_B.item()
            })

        # end of epoch: save checkpoint and sample images
        save_dict = {
            'epoch': epoch,
            'netG_A2B': netG_A2B.state_dict(),
            'netG_B2A': netG_B2A.state_dict(),
            'netD_A': netD_A.state_dict(),
            'netD_B': netD_B.state_dict(),
            'optG': optG.state_dict(),
            'optD_A': optD_A.state_dict(),
            'optD_B': optD_B.state_dict()
        }
        utils.save_checkpoint(save_dict, args.checkpoint_dir, epoch)

        # optionally: save some sample images
        try:
            utils.save_samples(real_A.cpu(), real_B.cpu(), fake_B.cpu(), fake_A.cpu(), out_dir=args.samples_dir, epoch=epoch)
        except Exception as e:
            print("Could not save samples:", e)

        schedulerG.step()
        schedulerD_A.step()
        schedulerD_B.step()

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/kaggle/working/dataset', help='root dir with trainA/trainB')
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/checkpoints', help='where to save models')
    parser.add_argument('--samples_dir', type=str, default='/kaggle/working/samples', help='where to save sample images')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--load_size', type=int, default=286)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--n_epochs', type=int, default=100, help='epochs with constant lr')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='epochs for linear lr decay')
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_id', type=float, default=0.5)
    args = parser.parse_args()
    train(args)
