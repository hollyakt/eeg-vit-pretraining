import math
from timm.utils import accuracy
from typing import Iterable, Optional
import sys
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import gc

from pathlib import Path

from timm.data import Mixup

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from torch.utils.data import DataLoader
    
    # Import your transform or define it
from torchvision import transforms
import vision_transformer as vits
#from datasets.ECoG_traintest import ECoGDataset_train, ECoGDataset_test, DataAugmentation_finetune
from datasets.ECOG90S_dataloader import EEG_Windows_Model2
import utils
from sklearn import metrics
import os


def get_args_parser():
    parser = argparse.ArgumentParser(
        'DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=101, type=int)
    parser.add_argument('--in_chans', default=128, type=int)
    parser.add_argument('--negative_class_mode', default=0, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')
    parser.add_argument('--ensemble', default=10, type=int,
                        help='aggregate over the last n epochs')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument(
        '--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float,
                        default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='/home/pmi_lab/EEG_challenge/R1_mini_L100/transformer/channel_wise/checkpoints/jbest_ckpt_ep0310.pth', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_location', default='/home/pmi_lab/EEG_challenge/EEG_challenge/spectrograms_5fold',
                        type=str, help='semantic granularity')
    parser.add_argument('--pfactor_csv', default='/home/pmi_lab/EEG_challenge/EEG_challenge/transformer/channel_wise/all_participants.csv',
                        type=str, help='path to p-factor csv file')

    parser.add_argument('--output_dir', default=R'/home/pmi_lab/EEG_challenge/EEG_challenge/finetuning/model2/f1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)

    parser.add_argument('--evaluate_every', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--train_folds', default='0', type=str,
                    help='Comma-separated fold indices for training (e.g., "0,2,3,4")')
    parser.add_argument('--val_fold', default='1', type=str,
                    help='Single fold index for validation (e.g., "1")')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    return parser


def main_train(args):
    #utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train = ECOG90S_npz(data_location=os.path.join(args.data_location,'train'),
    #                                   in_chans=args.in_chans,
    #                                   transform=DataAugmentation_finetune(True, args.input_size, in_chans=args.in_chans))
    
    # dataset_test =  ECOG90S_npz(data_location=os.path.join(args.data_location,'val'),                                                 
    #                                              in_chans=args.in_chans,
    #                                              transform=DataAugmentation_finetune(False, args.input_size,in_chans=args.in_chans ))
   
    # args.nb_classes = 2 #if args.negative_class_mode == 5 else 2

    args.train_folds = '0'
    args.val_fold = '1'
    
    train_fold_list = [f"f{x}" for x in args.train_folds.split(',')]
    val_fold = f"f{args.val_fold}"

    print(f"Training folds: {train_fold_list}")
    print(f"Validation fold: {val_fold}")

    # Create datasets for multiple training folds
    train_datasets = []
    for fold in train_fold_list:
        fold_dataset = EEG_Windows_Model2(
            data_location=os.path.join(args.data_location, fold),
            pfactor_csv_path=args.pfactor_csv,
            in_chans=args.in_chans,
            transform=None,
            normalize_pfactor=False,

        )
        train_datasets.append(fold_dataset)

    # Concatenate all training folds
    from torch.utils.data import ConcatDataset
    dataset_train = ConcatDataset(train_datasets)
    
    # dataset_train = EEG_Windows_Regression_Lazy(
    # data_location=os.path.join(args.data_location, 'ftrain'),
    # in_chans=args.in_chans,
    # transform=None,
    # require_rt=True,
    # normalize_rt=True
    # )

    dataset_test = EEG_Windows_Model2(
        data_location=os.path.join(args.data_location, val_fold),
        pfactor_csv_path=args.pfactor_csv,
        in_chans=args.in_chans,
        transform=None,
        normalize_pfactor=False,
    )
    

    args.nb_classes = 1  # Single output for regression
    args.task = 'regression'  # Mark as regression task
    
    # Get RT statistics from training set
    # rt_stats = dataset_train.get_rt_stats()
    # print(f"Training RT statistics: {rt_stats}")

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    gc.collect()
    torch.cuda.empty_cache()

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    dataset_loader_test = DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    # dataset_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, sampler=sampler_test,
    #     batch_size=int(1.5 * args.batch_size),
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False
    # )

    mixup_fn = None

    print(f"Creating model: {args.model}")

    model = vits.__dict__[args.model](num_classes=args.nb_classes, img_size=[args.input_size], drop_path_rate=args.drop_path,
                                      in_chans=args.in_chans)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    args.distributed = False
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss() #for regression

    output_dir = Path(args.output_dir)
    if args.finetune and Path(args.finetune).is_file():

        checkpoint = torch.load(args.finetune, map_location='cpu')

        state_dic = checkpoint['SiT_model'] if not 'MCSSL' in args.finetune else checkpoint['model']
        state_dic = {k.replace("module.", ""): v for k, v in state_dic.items()}

        state_dic = {k.replace("backbone.", ""): v for k, v in state_dic.items()}

        # interpolate position embedding
        
        pos_embed_checkpoint = state_dic['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int(
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size,
                                        orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dic['pos_embed'] = new_pos_embed

        msg = model_without_ddp.load_state_dict(state_dic, strict=False)
        print(msg)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    Best_test = [] #Best_10_RSNA_RICORD, Best_10_CNH = [], [], []

    best_val_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            args=args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_latest.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        # if epoch % args.evaluate_every == 0:
        #     test_stats, preds, targets = evaluate(
        #         dataset_loader_test, model, device)
        #     targets, preds = torch.cat(
        #         targets).numpy(), torch.cat(preds).numpy()
        #     Best_test.append(preds)

        #     l = len(Best_test)
        #     if l > args.ensemble:
        #         Best_test = Best_test[(
        #             l-args.ensemble):l]

        #     final_pred = 0
        #     for xx in Best_test:
        #         final_pred = final_pred + xx

        #     final_pred /= len(Best_test)
        #     if args.nb_classes == 2:
        #         log_stats = utils.store_results_cls_2(
        #             targets, final_pred, epoch, n_parameters, train_stats, test_stats, dataset_loader_test, ds= 'test')
        #     # elif args.nb_classes > 2:
        #     #     log_stats = utils.store_results_cls(
        #     #         targets, final_pred, epoch, n_parameters, train_stats, test_stats, dataset_loader_test)

        #     if args.output_dir and utils.is_main_process():
        #         with (output_dir / "log_test.txt").open("a") as f:
        #             f.write(json.dumps(log_stats) + "\n")

        if epoch % args.evaluate_every == 0:
            test_stats, preds, targets = evaluate(
                dataset_loader_test, model, device)
            targets, preds = torch.cat(
                targets).numpy(), torch.cat(preds).numpy()
            Best_test.append(preds)

            current_val_loss = test_stats.get('loss', float('inf'))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"Saving best model with val_loss: {best_val_loss:.4f}")
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                }, output_dir / 'best_model.pth')

            l = len(Best_test)
            if l > args.ensemble:
                Best_test = Best_test[(
                    l-args.ensemble):l]

            final_pred = 0
            for xx in Best_test:
                final_pred = final_pred + xx

            final_pred /= len(Best_test)
            
            # For regression task
            if args.nb_classes == 1:
                # Calculate regression metrics
                mse = np.mean((targets - final_pred) ** 2)
                mae = np.mean(np.abs(targets - final_pred))
                corr = np.corrcoef(targets.flatten(), final_pred.flatten())[0, 1]
                
                log_stats = {
                    'epoch': epoch,
                    'n_parameters': n_parameters,
                    'train_loss': train_stats.get('loss', 0),
                    'test_loss': test_stats.get('loss', 0),
                    'test_mse': float(mse),
                    'test_mae': float(mae),
                    'test_correlation': float(corr) if not np.isnan(corr) else 0.0,
                }
                
                print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, Correlation: {corr:.4f}")
                
            elif args.nb_classes == 2:
                log_stats = utils.store_results_cls_2(
                    targets, final_pred, epoch, n_parameters, train_stats, test_stats, dataset_loader_test, ds='test')
            else:
                # Fallback for other cases
                log_stats = {
                    'epoch': epoch,
                    'n_parameters': n_parameters,
                    'train_loss': train_stats.get('loss', 0),
                    'test_loss': test_stats.get('loss', 0),
                }

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log_test.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # outputs = model(samples, classify=True)
            # loss = criterion(outputs, targets)
            outputs = model(samples, classify=True)
            # Ensure both have same shape [16, 1] -> [16, 1] 
            if outputs.shape != targets.shape:
                outputs = outputs.view(-1)  # [16, 1] -> [16]
                targets = targets.view(-1)   # [16, 1] -> [16] 
            loss = criterion(outputs, targets)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss() #for regression
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    targets, preds = [], []
    # for images, target in metric_logger.log_every(data_loader, 100, header):
    #     images = images.to(device, non_blocking=True)
    #     target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        #     output = model(images, classify=True)
        #     loss = criterion(output, target)

        # preds.append(output.cpu().detach())
        # targets.append(target.cpu().detach())

    for images, target in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images, classify=True)
            # Make shapes match
            if output.shape != target.shape:
                output = output.view(-1)
                target = target.view(-1)
            loss = criterion(output, target)

        preds.append(output.cpu().detach())
        targets.append(target.cpu().detach())

        #acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        #metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print('* Loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, preds, targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Finetuning on Downstream script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_train(args)
