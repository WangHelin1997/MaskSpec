import math
import sys
from typing import Iterable
import torch
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.misc as misc
import utils.lr_sched as lr_sched
from timm.utils import accuracy
import numpy as np
from sklearn import metrics
from torch.nn import functional as F

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        targets_mask = targets[:, 20:]
        targets = targets[:, :20]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_mask = targets_mask.to(device, non_blocking=True)
        targets = targets > 0.5
        targets = targets.float()

        with torch.cuda.amp.autocast():
            outputs, targets = model(samples, targets, False, None)
            samples_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
            samples_loss = targets_mask * samples_loss
            loss = samples_loss.mean()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Eval:'

    # switch to evaluation mode
    model.eval()
    alloutput = []
    alltarget = []
    allmask = []
    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]
        targets_mask = target[:, 20:]
        target = target[:, :20]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        targets_mask = targets_mask.to(device, non_blocking=True)
        target = target > 0.5
        target = target.float()

        # compute output
        with torch.cuda.amp.autocast():
            output, _ = model(images, target, specmix=False)
            loss = F.binary_cross_entropy_with_logits(output, target, reduction="none")
            loss = targets_mask * loss
        output = torch.sigmoid(output).cpu().detach()
        target = target.cpu().detach()
        targets_mask = targets_mask.cpu().detach()
        metric_logger.update(loss=loss.mean().item())
        alloutput.append(output)
        alltarget.append(target)
        allmask.append(targets_mask)
    # gather the stats from all processes
    alloutput = np.concatenate(alloutput, 0)
    alltarget = np.concatenate(alltarget, 0)
    allmask = np.concatenate(allmask, 0)
    average_precision = np.array([
        metrics.average_precision_score(
            alltarget[:, i], alloutput[:, i], sample_weight=allmask[:, i]) 
            for i in range(alloutput.shape[1])])

    auc = np.array([
         metrics.roc_auc_score(
            alltarget[:, i], alloutput[:, i], sample_weight=allmask[:, i]) 
            for i in range(alloutput.shape[1])])

    metric_logger.meters['mAP'].update(average_precision.mean().item(), n=alltarget.shape[0])
    metric_logger.meters['mAUC'].update(auc.mean().item(), n=alltarget.shape[0])
    metric_logger.synchronize_between_processes()
    print('* mAP {map.global_avg:.3f} mAUC {mauc.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(map=metric_logger.mAP, mauc=metric_logger.mAUC, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
