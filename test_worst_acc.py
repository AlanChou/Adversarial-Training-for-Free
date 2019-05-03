'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import argparse

from tqdm import tqdm
import numpy as np
from models.wideresnet import *
from utils import *
from advertorch.attacks import LinfPGDAttack


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--seed', default=11111, type=int, help='seed')
parser.add_argument('--epoch', default=0, type=int, help='load checkpoint from that epoch')
parser.add_argument('--model', default='wideresnet', type=str)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--iteration', default=100, type=int)
parser.add_argument('--epsilon', default=8./255, type=float)
parser.add_argument('--step_size', default=2./255, type=float)


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])


testset = torchvision.datasets.CIFAR10(root='/home/hsinpingchou/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

print('==> Building model {}..'.format(args.model))
if args.model == 'wideresnet':
    net = WideResNet_28_10()
else:
    raise ValueError('No such model.')

def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)   
            with torch.enable_grad(): 
                adv = adversary.perturb(inputs, targets)
            outputs = net(adv)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))


    # Save checkpoint.
    acc = 100.*correct/total
    print('Test Acc of ckpt.{}: {}'.format(args.epoch, acc))


print('==> Loading from checkpoint epoch {}..'.format(args.epoch))
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.{}'.format(args.epoch))
net.load_state_dict(checkpoint['net'])
net = net.to(device)
net.eval()


adversary = LinfPGDAttack(
    net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon,
    nb_iter=args.iteration, eps_iter=args.step_size, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

test(adversary)

