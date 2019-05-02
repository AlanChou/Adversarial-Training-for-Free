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
from models import *
import random
import numpy as np
from advertorch.attacks import LinfPGDAttack
from models.wideresnet import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--seed', default=11111, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--epsilon', default=8.0/255, type=float)
parser.add_argument('--m', default=8, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Normalization messes with l-inf bounds.
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)


print('==> Building model..')
net = WideResNet_28_10()
epsilon = args.epsilon
m = args.m
delta = torch.zeros(args.train_batch_size, 3, 32, 32)
delta = delta.to(device)
net = net.to(device)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.best')
    basic_net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)

adversary = LinfPGDAttack(
    net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8./255.,
    nb_iter=100, eps_iter=2./255., rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    global delta

    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        for i in range(m):
            optimizer.zero_grad()
            adv = (inputs+delta).detach()
            adv.requires_grad_()
            outputs = net(adv)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            grad = adv.grad.data
            delta = delta + epsilon * torch.sign(grad)
            delta = torch.clamp(delta, -epsilon, epsilon)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    acc = 100.*correct/total
    print('Train acc:', acc)

    
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.{}'.format(epoch))
    best_acc = acc

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)   
            with torch.enable_grad(): 
                adv = adversary.perturb(inputs, targets)
            outputs = net(adv)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))


    # Save checkpoint.
    acc = 100.*correct/total
    print('Val acc:', acc)
    if acc > best_acc:
        print('Saving best..')
        state = {
            'net': net.state_dict(),
            'best_acc': acc,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.best')
        best_acc = acc

def adjust_learning_rate(optimizer, epoch):
    if epoch < 15:
        lr = 0.1
    elif epoch >= 15 and epoch <23:
        lr = 0.01
    elif epoch >= 23:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(start_epoch, 31):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    if epoch > 20:
        test(epoch)
