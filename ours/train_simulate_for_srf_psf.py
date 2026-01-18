import time
import os
import torch
from torch.utils.data import DataLoader
from GetDataSet import MakeSimulateDataset
import torch.nn as nn
from model import transform
import copy
import argparse
import json
import cv2
import numpy
import utils
import random, tqdm
import quality_index
import math
import ssim

class Res_block(nn.Module):
    def __init__(self, num_channels=64, kernel_size=3):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = x + self.conv2(self.relu(self.conv1(x)))
        return y

class DemoisacNet(nn.Module):
    def __init__(self, args):
        super(DemoisacNet, self).__init__()
        self.D = 6
        self.channels = 64

        self.conv_head1 = nn.Conv2d(args.num_bands+1, self.channels, 3, 1, 1)
        self.conv_head2 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.res_blks = nn.ModuleList([Res_block(self.channels) for _ in range(self.D)])
        self.conv_tail = nn.Conv2d(self.channels, args.num_bands, 3, 1, 1)

        self.relu = nn.ReLU()
    
    def forward(self, mosaic, pan):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bilinear")
        x = torch.cat((mosaic_interpolate, pan), 1)
        x = self.conv_head2(self.relu(self.conv_head1(x)))
        for i in range(self.D):
            x = self.res_blks[i](x)
        y = self.conv_tail(x)
        return y

class Degrade_DK(nn.Module):
    def __init__(self, ksize=13):
        super(Degrade_DK, self).__init__()

        self.psf = nn.Conv2d(1, 1, ksize, 1, ksize//2, bias=False)
    
    def forward(self, x):
        if x.shape[1] > 1:
            z = []
            for i in range(x.shape[1]):
                z.append(self.psf(x[:, i:i+1, :, :]))
            z = torch.cat(z, 1)
        else:
            z = self.psf(x)

        z = torch.nn.functional.avg_pool2d(z, kernel_size=2, stride=2)
        return z

class Degrade_R(nn.Module):
    def __init__(self, args):
        super(Degrade_R, self).__init__()
        self.D = 3
        self.channels = 64
        self.msfa_size = [4, 4]

        self.spec_res = nn.Conv2d(args.num_bands, 1, 1, 1, 0, bias=False)
    
    def forward(self, x):
        y = self.spec_res(x)
        return y

def train(demosaic_net: nn.Module, degrade_dk: nn.Module, degrade_r: nn.Module, optimizers, train_dataloader, val_dataloader, args):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    t = time.time()
    device = next(degrade_dk.parameters()).device
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0], MSFA.shape[1]).to(device)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i, j] = 1
    best_epoch, best_psnr = 0, 0
    numpy.set_printoptions(precision=3, suppress=True)
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        demosaic_net.train()
        degrade_dk.train()
        degrade_r.train()
        start_time = time.time()

        loss_per_epoch = 0
        for cnt, data in enumerate(tqdm.tqdm(train_dataloader)):
            z, y = data[0].to(device), data[1].to(device)

            xlr_first = demosaic_net(z, y)

            Rxlr = degrade_r(xlr_first)
            z_degrade = torch.nn.functional.conv2d(xlr_first, msfa_kernel, stride = args.msfa_size, groups = xlr_first.shape[1])
            z_degrade = torch.nn.functional.pixel_shuffle(z_degrade, upscale_factor=4)

            loss_oc = torch.nn.functional.mse_loss(z, z_degrade)

            xlr_first = transform(xlr_first, msfa_size=4, spatial_ratio=2).detach()

            z_degrade = torch.nn.functional.conv2d(xlr_first, msfa_kernel, stride = args.msfa_size, groups = xlr_first.shape[1])
            z_degrade = torch.nn.functional.pixel_shuffle(z_degrade, upscale_factor=4)

            xlr_second = demosaic_net(z_degrade, y)

            loss_ei = torch.nn.functional.mse_loss(xlr_second, xlr_first)

            yDK = degrade_dk(y)

            loss_eq = torch.nn.functional.mse_loss(Rxlr, yDK)

            loss = loss_oc + loss_ei + loss_eq

            optimizers[0].zero_grad()
            optimizers[1].zero_grad()
            optimizers[2].zero_grad()
            loss.backward()
            optimizers[0].step()
            optimizers[1].step()
            optimizers[2].step()

            for name, param in degrade_dk.named_parameters():
                param.requires_grad = False
                param[param<0] = 0
                param /= param.sum()
                param.requires_grad = True
            
            for name, param in degrade_r.named_parameters():
                if "spec_res" in name:
                    param.requires_grad = False
                    param[param<0] = 0
                    param.requires_grad = True

            loss_per_epoch += loss.detach().item()
          
        loss_per_epoch /= cnt + 1
        # val
        psnr_avg = 0.
        degrade_dk.eval()
        degrade_r.eval()
        with torch.no_grad():
            for cnt, data in enumerate(val_dataloader):
                z, y = data[0].to(device), data[1].to(device)
                xlr = demosaic_net(z, y)
                Rxlr = degrade_r(xlr)
                yDK = degrade_dk(y)

                psnr_avg += quality_index.calc_psnr(Rxlr, yDK).item()

        psnr_avg /= cnt+1

        if args.record is not False:
            record = []
            if os.path.exists(args.record):
                with open(args.record, "r") as f:
                    record = json.load(f)
            record.append({"epoch": epoch,
                           "loss": loss_per_epoch,
                           "psnr": psnr_avg,
                           "best_psnr": best_psnr,
                           "best_epoch": best_epoch,
                           "learning rate": optimizers[0].param_groups[0]["lr"]
                           })
            with open(args.record, "w") as f:
                record = json.dump(record, f, indent=2)

        # save model with highest PSNR
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            if best_epoch != 0:
                os.remove(os.path.join(args.dir_model, "best_{}.pth".format(best_epoch)))
            best_epoch = epoch
            torch.save({"demosaic_net": demosaic_net.state_dict(), 
                        "degrade_dk": degrade_dk.state_dict(), 
                        "degrade_r": degrade_r.state_dict()}, os.path.join(args.dir_model, "best_{}.pth".format(epoch)))
        
        # save model at some frequency
        if epoch % args.save_freq == 0:
            torch.save({"demosaic_net": demosaic_net.state_dict(), 
                        "degrade_dk": degrade_dk.state_dict(), 
                        "degrade_r": degrade_r.state_dict()}, os.path.join(args.dir_model, f"{epoch}.pth"))

        # log
        print("Epoch: ", epoch,
            "loss: %.4f"%loss_per_epoch,
            "time: %.2f"%((time.time() - start_time) / 60), "min",
            "loss: %.4f"%loss_per_epoch,
            "psnr: %.4f"%psnr_avg,
            "best_psnr: %.4f"%best_psnr,
            "best_epoch: ", best_epoch,
            "learning rate: ", optimizers[0].param_groups[0]["lr"], "\n",
            )
        loss_per_epoch = 0 

    print(f"Total time: {(time.time() - t) / 60} min")
    print("Best epoch: {}, Best PSNR: {}".format(best_epoch, best_psnr))

def main(args):
    dir_idx = os.path.join("./", str(args.idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)
    args.cache_path = dir_idx

    dir_model = os.path.join(dir_idx, "model")
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    args.dir_model = dir_model

    if args.record is True:
        dir_record = os.path.join(dir_idx, "record")
        if not os.path.exists(dir_record):
            os.makedirs(dir_record)
        args.dir_record = dir_record
        args.record = os.path.join(dir_record, "record.json")
        if args.resume == "" and os.path.exists(args.record):
            os.remove(args.record)

    total_iterations = args.epochs * args.iters_per_epoch
    print('total_iterations:{}'.format(total_iterations))

    train_set = MakeSimulateDataset(args, "train")
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_set = MakeSimulateDataset(args, "test")
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    demosaic_net = DemoisacNet(args)
    degrade_dk = Degrade_DK(13)
    degrade_r = Degrade_R(args)

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        demosaic_net.load_state_dict(torch.load(backup_pth)["demosaic_net"], strict=False)
        degrade_dk.load_state_dict(torch.load(backup_pth)["degrade_dk"], strict=False)
        degrade_r.load_state_dict(torch.load(backup_pth)["degrade_r"], strict=False)
    else:
        print('==> Train from scratch')
    
    demosaic_net = demosaic_net.to(f"cuda:{args.device}")
    degrade_dk = degrade_dk.to(f"cuda:{args.device}")
    degrade_r = degrade_r.to(f"cuda:{args.device}")

    optimizer0 = torch.optim.Adam(demosaic_net.parameters(), args.lr)
    optimizer1 = torch.optim.Adam(degrade_dk.parameters(), args.lr)
    optimizer2 = torch.optim.Adam(degrade_r.parameters(), args.lr)

    train(demosaic_net, degrade_dk, degrade_r, [optimizer0, optimizer1, optimizer2], train_dataloader, val_dataloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--idx', type=int, default=1, help='Index to identify models.')
    parser.add_argument('--dataset', type=str, default="Ours", help='Dataset to be loaded.')
    parser.add_argument('--train_size', type=int, default=128, help='Size of the training image in a batch.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--noise_level', nargs="+", type=float, default=[0.01, 0.05], help='Noise level when generating simulated data.')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    parser.add_argument('--stride', type=int, default=64, help='Stride when crop an original image into patches.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training dataset.')
    parser.add_argument('--epochs', type=int, default=1000, help='Total epochs to train the model.')
    parser.add_argument('--iters_per_epoch', type=int, default=100, help='Iteration steps per epoch.')
    parser.add_argument('--save_freq', type=int, default=50, help='Save the checkpoints of the model every [save_freq] epochs.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer.')
    parser.add_argument('--lr_decay', action="store_true", help='Determine if to decay the learning rate.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to train the model.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')
    parser.add_argument('--num_workers', type=int, default=1, help='Num_workers to train the model.')
    parser.add_argument('--resume', type=str, default='', help='Index of the model to be resumed, eg. 1000.')
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')

    args = parser.parse_args()
    main(args)