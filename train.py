from __future__ import print_function
import argparse
import os
import wandb
from tqdm import tqdm
from math import log10
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--root_dir', type=str, default='', help='root directory of project')
parser.add_argument('--experiments_dir', type=str, default='experiments', help='experiments path in root dir')
parser.add_argument('--project_name', type=str, default='', help='project name')

parser.add_argument('--ignore-wandb', action='store_true', help='flag to ignore wandb logging')
parser.add_argument('--wandb-project', default='palette_scene2scene', type=str, help='wandb project to use')
    
parser.add_argument('--train_dataset_input_dir', type=str, default='', help='directory to train dataset: from')
parser.add_argument('--train_dataset_target_dir', type=str, default='', help='directory to train dataset: to')
parser.add_argument('--test_dataset_input_dir', type=str, default='', help='directory to test dataset: from')
parser.add_argument('--test_dataset_target_dir', type=str, default='', help='directory to test dataset: to')

parser.add_argument('--dataset', type=str, default='facades', help='dataset_name')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help="a2b or b2a. b2a inverts dataset's input and target")
parser.add_argument('--image_size', default=[512, 512], nargs='+', help='resize image to [h, w]')
parser.add_argument('--patch_size', default=[256, 256], nargs='+', help='resize patches to size [h, w]')

parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default = True, help='use cuda?')
parser.add_argument('--gpu_id', type=int, default=0, gelp='ID odf GPU to use')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')

#opt = parser.parse_args()
opt, unknown = parser.parse_known_args()
now = datetime.now()
opt.project_name = opt.project_name + f'__{str(now.strftime("%d_%m_%Y__%H%M%S"))}'
print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device(f"cuda:{opt.gpu_id}" if opt.cuda else "cpu")

# tensorboard log
if not opt.ignore_wandb:
    wandb.init(project=opt.wandb_project, sync_tensorboard=True, name=opt.name)
if not os.path.exists(os.path.join(opt.root_dir, opt.experiments_dir, opt.project_name, 'logs')):
    os.makedirs(os.path.join(opt.root_dir, opt.experiments_dir, opt.project_name, 'logs'))
writer = SummaryWriter(log_dir=os.path.join(opt.root_dir, opt.experiments_dir, opt.project_name, 'logs'))
    

print('===> Loading datasets')
root_path = opt.root_dir
train_set = get_training_set(opt.train_dataset_input_dir, 
                             opt.train_dataset_target_dir, 
                             opt.direction,
                             opt.image_size,
                             opt.patch_size)
test_set = get_test_set(opt.test_dataset_input_dir, 
                        opt.test_dataset_target_dir, 
                        opt.direction,
                        opt.image_size, 
                        opt.patch_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

highest_psnr = 0
filename_highest_psnr_checkpoint = ""

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)


# train
print("===> Starting Training")
for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1), total=len(range(opt.epoch_count, opt.niter + opt.niter_decay + 1))):
    print(f"Running Training round for epoch {epoch}")
    for iteration, batch in tqdm(enumerate(training_data_loader, 1), total=len(training_data_loader)):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()

        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb

        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
        writer.add_scalar('Train/epoch', epoch, iteration)
        writer.add_scalar('Train/loss_d_fake', np.mean(loss_d_fake.cpu().numpy()), iteration)
        writer.add_scalar('Train/loss_d_real', np.mean(loss_d_real.cpu().numpy()), iteration)
        writer.add_scalar('Train/loss_d', np.mean(loss_d.cpu().numpy()), iteration)
        writer.add_scalar('Train/loss_g_gan', np.mean(loss_g_gan.cpu().numpy()), iteration)
        writer.add_scalar('Train/loss_g_L1', np.mean(loss_g_l1.cpu().numpy()), iteration)
        writer.add_scalar('Train/loss_g', np.mean(loss_g.cpu().numpy()), iteration)
    
    writer.add_scalar('Train/LR_g', optimizer_g.param_groups[0]['lr'], epoch)
    writer.add_scalar('Train/LR_d', optimizer_d.param_groups[0]['lr'], epoch)

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    psnr_sum = 0
    last_prediction = None
    print(f"Running Validation round for epoch {epoch}")
    for batch in tqdm(testing_data_loader, total=len(testing_data_loader)):
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        psnr_sum += psnr
        last_prediction = prediction.cpu()

    avg_psnr = psnr_sum / len(testing_data_loader)

    if avg_psnr > highest_psnr:
          highest_psnr = avg_psnr

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    writer.add_scalar('Val/psnr', avg_psnr, epoch)
    writer.add_scalar('Val/psnr_highest', highest_psnr, epoch)
    outputs_grid = torchvision.utils.make_grid([last_prediction[idx] for idx in range(len(last_prediction))])
    writer.add_image(tag='Val/log_images', 
                    img_tensor=outputs_grid,
                    global_step = epoch,
                    dataformats = 'CHW')


    #checkpoint
    if epoch % 50 == 0:
        if not os.path.exists(os.path.join(opt.root_dir, opt.experiments_dir, opt.project_name, 'weights')):
            os.makedirs(os.path.join(opt.root_dir, opt.experiments_dir, opt.project_name, 'weights'))
        net_g_model_out_path = f"{opt.root_dir}/{opt.experiments_dir}/{opt.project_name}/weights/netG_model_epoch_{epoch}.pth"
        # filename_highest_psnr_checkpoint = net_g_model_out_path
        #net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        print(f"Checkpoint saved to {opt.root_dir}/{opt.experiments_dir}/{opt.project_name}/weights")

    if avg_psnr >= highest_psnr:
        if not os.path.exists(os.path.join(opt.root_dir, opt.experiments_dir, opt.project_name, 'weights')):
            os.makedirs(os.path.join(opt.root_dir, opt.experiments_dir, opt.project_name, 'weights'))
        net_g_model_out_path_best = f"{opt.root_dir}/{opt.experiments_dir}/{opt.project_name}/weights/netG_model_highest_psnr_{avg_psnr}_epoch_{epoch}.pth"
        torch.save(net_g, net_g_model_out_path_best)
        filename_highest_psnr_checkpoint = net_g_model_out_path
        print(f"Highest psnr checkpoint saved to {opt.root_dir}/{opt.experiments_dir}/{opt.project_name}/weights")
        #torch.save(net_d, net_d_model_out_path)

print("The highest psnr is {} and the checkpoint is {}".format(avg_psnr, filename_highest_psnr_checkpoint))
print("Training Ended with {}/{} epochs complete".format(epoch, opt.niter + opt.niter_decay))

writer.close()
if not opt.ignore_wandb:
    wandb.finish()
print("ALL DONE")