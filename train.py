import torch
from util.torch_dist_sum import *
from data.imagenet import *
from data.augmentation import *
import torch.nn as nn
from util.meter import *
from network.simclr import Simclr
import time
from util.accuracy import accuracy
import apex
from math import sqrt
import math
from util.LARS import LARS


epochs = 200
warm_up = 10


def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img1, img2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        data_time.update(time.time() - end)

        if local_rank is not None:
            img1 = img1.cuda(local_rank, non_blocking=True)
            img2 = img2.cuda(local_rank, non_blocking=True)


        # compute output
        output, target = model(img1, img2, rank)
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img1.size(0))
        top1.update(acc1[0], img1.size(0))
        top5.update(acc5[0], img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 and rank == 0:
            progress.display(i)


def main():
    from torch.nn.parallel import DistributedDataParallel
    from util.dist_init import dist_init
    
    
    rank, local_rank, world_size = dist_init()
    batch_size = 128 # single gpu
    num_workers = 4

    base_lr = 0.075 * sqrt(batch_size * world_size)

    model = Simclr()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    
    
    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, 'ignore': True },
                                {'params': rest_params, 'weight_decay': 1e-6, 'ignore': False}], 
                                lr=base_lr, momentum=0.9, weight_decay=1e-6)

    optimizer = LARS(optimizer, eps=0.0)
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
    model = DistributedDataParallel(model, device_ids=[local_rank])

    torch.backends.cudnn.benchmark = True


    # feel free to change your dataset setup
    # use weak augmentation for warmup
    weak_aug_train_dataset = ImagenetContrastive(aug=weak_contrastive_aug)
    weak_aug_train_sampler = torch.utils.data.distributed.DistributedSampler(weak_aug_train_dataset)
    weak_aug_train_loader = torch.utils.data.DataLoader(
        weak_aug_train_dataset, batch_size=batch_size, shuffle=(weak_aug_train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=weak_aug_train_sampler, drop_last=True)


    # switch to simclr augmentation
    train_dataset = ImagenetContrastive(aug=contrastive_aug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    iteration_per_epoch = train_loader.__len__()
    criterion = nn.CrossEntropyLoss().cuda(local_rank)


    model.train()
    for epoch in range(epochs):
        if epoch < warm_up:
            weak_aug_train_sampler.set_epoch(epoch)
            train(weak_aug_train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr)
        else:
            train_sampler.set_epoch(epoch)
            train(train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr)
    
        if rank == 0 and (epoch + 1) % 50 ==  0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': apex.amp.state_dict(),
            }, 'checkpoints/simclr_apex{}.pth'.format(epoch+1))


if __name__ == "__main__":
    main()

