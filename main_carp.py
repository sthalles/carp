import argparse
import os
import shutil
import sys
import datetime
import time
import math
import yaml

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from modules.carp_head import CARPHead
from modules.carp_loss import CARPLoss
from modules.random_partition import RandomPartition
from modules.view_generator import ViewGenerator
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
            acc1, acc5 = utils.accuracy(student_output[0][0], torch.argmax(
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CARP', parents=[get_args_parser()])
    args = parser.parse_args()
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_carp(args)
