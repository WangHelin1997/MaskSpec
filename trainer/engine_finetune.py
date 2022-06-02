import math
import sys
from typing import Iterable
from sklearn import metrics
import torch
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.misc as misc
import utils.lr_sched as lr_sched
from timm.loss import BinaryCrossEntropy
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
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

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs, targets = model(samples, targets, args.specmix, args.mixup_alpha)
            loss = criterion(outputs, targets)

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
    criterion = BinaryCrossEntropy(smoothing=0.0, reduction='mean')

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    alloutput = []
    alltarget = []
    for batch in metric_logger.log_every(data_loader, 200, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, _ = model(images, target, specmix=False)
            loss = criterion(output, target)
        
        output = torch.sigmoid(output).cpu().detach()
        target = target.cpu().detach()
        metric_logger.update(loss=loss.item())
        alloutput.append(output)
        alltarget.append(target)
    # gather the stats from all processes
    alloutput = np.concatenate(alloutput, 0)
    alltarget = np.concatenate(alltarget, 0)
    average_precision = metrics.average_precision_score(
            alltarget, alloutput, average='macro')
    auc = metrics.roc_auc_score(alltarget, alloutput, average='macro')
    metric_logger.meters['mAP'].update(average_precision.item(), n=alltarget.shape[0])
    metric_logger.meters['mAUC'].update(auc.item(), n=alltarget.shape[0])
    metric_logger.synchronize_between_processes()
    print('* mAP {map.global_avg:.3f} mAUC {mauc.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(map=metric_logger.mAP, mauc=metric_logger.mAUC, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
