import argparse
from collections import defaultdict
import os
import shutil
import sys
import datetime
import time
import math
import json
from pathlib import Path

import yaml

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
import utils


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('CARP', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=torchvision_archs,
                        help="""Name of architecture to train.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the CARP head output.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the CARP head.""")
    parser.add_argument('--momentum_teacher', default=0.99, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--use_bn_in_head', default=True, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: True)")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training.""")
    parser.add_argument('--weight_decay', type=float, default=0.000001, help="""Initial value of the
        weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.000001, help="""Final value of the
        weight decay. We use a cosine schedule for WD.""")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="""We use gradient 
                        accumulation to simulate large batch sizes in small gpus.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.45, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=0, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=0.0048, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='lars', type=str,
                        choices=['lars'], help="""Type of optimizer.""")
    parser.add_argument("--partition_size", default=512, type=int,
                        help="The number of random prototypes in a partition.")
    parser.add_argument("--bottleneck_dim", default=256,
                        type=int, help="Dimensionality of the embedding vector.")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.2, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping.""")
    parser.add_argument('--local_crops_number', type=int, default=6, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.2),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='../../../../../../data/ImageNet2012/train', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--resume_from_dir', default=".",
                        type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=25, type=int,
                        help='Save checkpoint every x epochs.')
    parser.add_argument('--print_freq', default=100, type=int,
                        help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    return parser


def train_carp(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = ViewGenerator(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    if args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[
            args.arch](zero_init_residual=True)
        teacher = torchvision_models.__dict__[
            args.arch](zero_init_residual=True)
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, CARPHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        bottleneck_dim=args.bottleneck_dim
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        CARPHead(embed_dim, args.out_dim, args.use_bn_in_head,
                 bottleneck_dim=args.bottleneck_dim),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # total number of crops = 2 global crops + local_crops_number
    args.ncrops = args.local_crops_number + 2

    # ============ preparing loss ... ============
    criterion = CARPLoss()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    # to use with convnet and large batches
    optimizer = utils.LARS(params_groups)

    # init optimizer
    optimizer.zero_grad()

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.gradient_accumulation_steps * args.batch_size_per_gpu *
                   utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.resume_from_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler
    )
    start_epoch = to_restore["epoch"]

    summary_writer = None
    if utils.is_main_process():
        summary_writer = SummaryWriter()
        shutil.copyfile(
            "./main_carp.py", os.path.join(summary_writer.log_dir,
                                           "main_carp.py")
        )
        shutil.copyfile(
            "./utils.py", os.path.join(summary_writer.log_dir, "utils.py")
        )
        stats_file = open(
            os.path.join(summary_writer.log_dir, "stats.txt"), "a", buffering=1
        )
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
        with open(os.path.join(summary_writer.log_dir, "metadata.txt"), "a") as f:
            yaml.dump(args, f, allow_unicode=True)
            f.write(str(student))
            f.write(str(teacher))

    random_partitioning = RandomPartition(args).cuda()

    start_time = time.time()
    print("Starting CARP training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of CARP ... ============
        train_one_epoch(student, teacher, teacher_without_ddp, criterion,
                        data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                        epoch, fp16_scaler, random_partitioning, summary_writer, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        if summary_writer is not None:
            utils.save_on_master(save_dict, os.path.join(
                summary_writer.log_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            if summary_writer is not None:
                utils.save_on_master(save_dict, os.path.join(
                    summary_writer.log_dir, f'checkpoint{epoch:04}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk
        ]


def train_one_epoch(student, teacher, teacher_without_ddp, criterion, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, random_partitioning, summary_writer, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (images, _) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        it = len(data_loader) * epoch + i  # global training iteration

        lr = lr_schedule[it]
        m = momentum_schedule[it]

        learning_rates.update(lr)
        sync_gradients = (
            (i + 1) % args.gradient_accumulation_steps == 0) or (i + 1 == len(data_loader))

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        if not sync_gradients:
            with student.no_sync():
                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    student_output = student(images)
                    # only the 2 global views pass through the teacher
                    teacher_output = teacher(images[:2])

                # Random Parition strategy
                student_output, teacher_output = random_partitioning(
                    student_output.float(), teacher_output.float(), args.partition_size)

                c, h = criterion(student_output, teacher_output)
                loss = c + h
                loss /= args.gradient_accumulation_steps
                # accumulate gradients
                fp16_scaler.scale(loss).backward()
        else:
            # update learning rate according to schedule
            for j, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr
                if j == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_output = student(images)
                # only the 2 global views pass through the teacher
                teacher_output = teacher(images[:2])

            # random Parition strategy
            student_output, teacher_output = random_partitioning(
                student_output.float(), teacher_output.float(), args.partition_size)

            c, h = criterion(student_output, teacher_output)
            loss = c + h
            loss /= args.gradient_accumulation_steps

            # EMA update for the teacher
            with torch.no_grad():
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            optimizer.zero_grad()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        losses.update(loss.item(), images[0].size(0))

        if summary_writer is not None and it % args.print_freq == 0:
            acc1, acc5 = accuracy(student_output[0][0], torch.argmax(
                teacher_output[1][0], dim=1), topk=(1, 5))
            summary_writer.add_scalar("loss/total", loss.item(), it)
            summary_writer.add_scalar("loss/consistency", c.item(), it)
            summary_writer.add_scalar("loss/entropy", h.item(), it)
            summary_writer.add_scalar("momentum", m, it)
            summary_writer.add_scalar("lr", lr, it)
            summary_writer.add_scalar("acc/top1", acc1, it)
            summary_writer.add_scalar("acc/top5", acc5, it)

            n_protos = student_output[0][0].shape[1]
            summary_writer.add_histogram(
                f"dist/probs/{n_protos}", torch.argmax(student_output[0][0], dim=1), it)
            summary_writer.add_histogram(
                f"dist/targets/{n_protos}", torch.argmax(teacher_output[1][0], dim=1), it)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


class CARPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def cluster_loss(p, q, EPS):
        # assert inputs.shape == targets.shape
        # assert inputs.requires_grad == True
        # assert targets.requires_grad == False

        loss = torch.einsum("knc,knc->kn", [p, q])
        loss = torch.clamp(loss, EPS, 1.0 - EPS)
        loss = -torch.log(loss).mean()
        return loss

    def forward(self, student_output, teacher_output):
        EPS = torch.finfo(student_output[0].dtype).eps
        consistency = 0
        count = 0
        for i in range(len(student_output)):
            for j in range(len(teacher_output)):
                if i == j:
                    continue
                consistency += self.cluster_loss(
                    student_output[i], teacher_output[j], EPS)
                count += 1

        consistency /= count

        p = torch.cat(student_output, dim=1)
        q = torch.cat(teacher_output, dim=1)
        probs = torch.cat([p, q], dim=1)  # [N_GROUPS, 2*BS, DIM]
        probs = torch.transpose(probs, 0, 1)  # [2*BS, N_GROUPS, DIM]
        probs = AllGather.apply(probs)

        entropy = self.kl_div(torch.mean(probs, dim=0), EPS)
        return consistency, entropy

    @staticmethod
    def kl_div(p, EPS):
        return (
            torch.log(torch.tensor(
                p.shape[-1], dtype=p.dtype, device=p.device))
            + torch.sum(p * torch.log(torch.clamp(p, EPS, 1.0 - EPS)), axis=-1)
        ).mean()


class CARPHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = self.last_layer(x)
        return x


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
                dist.is_available()
                and dist.is_initialized()
                and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x)
                       for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
                dist.is_available()
                and dist.is_initialized()
                and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * \
                (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ViewGenerator(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([utils.Solarize()], p=0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class RandomPartition(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ncrops = args.ncrops
        self.n_prototypes = args.out_dim
        self.weights = torch.ones([args.out_dim,], dtype=torch.float)

    def forward(self, student_output, teacher_output, partition_size):

        student_out = student_output.chunk(self.ncrops)
        teacher_out = teacher_output.detach().chunk(2)

        number_of_partitions = self.n_prototypes // partition_size

        # logic for rangom partioning into subgroups
        if utils.is_dist_avail_and_initialized():

            if utils.get_rank() == 0:
                rand_cluster_indices = torch.multinomial(
                    self.weights,
                    number_of_partitions * partition_size,
                    replacement=False,
                ).cuda()
            else:
                rand_cluster_indices = torch.zeros(
                    (number_of_partitions * partition_size), dtype=torch.long
                ).cuda()

            torch.distributed.broadcast(rand_cluster_indices, src=0)
        else:
            rand_cluster_indices = torch.multinomial(
                self.weights,
                number_of_partitions * partition_size,
                replacement=False,
            ).cuda()

        split_cluster_ids = torch.stack(
            torch.split(rand_cluster_indices, partition_size)
        )

        probs_list = []
        for log_view in student_out:
            predictions_group = self.get_logits_group(
                log_view, split_cluster_ids, partition_size)
            probs_list.append(predictions_group)

        targets_list = []
        for tar_view in teacher_out:
            targets_group = self.get_logits_group(
                tar_view, split_cluster_ids, partition_size)
            targets_list.append(targets_group)

        return probs_list, targets_list

    def get_logits_group(self, logits, split_cluster_ids, partition_size):
        logits_group = logits[:, split_cluster_ids.flatten()]
        logits_group = logits_group.split(partition_size, dim=1)
        # out shape [N_BLOCKS, BS, BLOCK_SIZE]
        logits = torch.stack(logits_group, dim=0)
        probs = torch.softmax(logits, dim=-1)
        return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CARP', parents=[get_args_parser()])
    args = parser.parse_args()
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_carp(args)
